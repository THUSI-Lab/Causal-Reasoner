from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


QUESTION_PATHS = [
    ("question",),
    ("query",),
    ("prompt",),
    ("input",),
    ("instruction",),
    ("qa", "question"),
    ("sample", "question"),
    ("meta", "question"),
]

ANSWER_PATHS = [
    ("answer",),
    ("response",),
    ("output",),
    ("target",),
    ("reference_answer",),
    ("gold_answer",),
    ("qa", "answer"),
    ("sample", "answer"),
    ("meta", "answer"),
]

IMAGE_KEYS = [
    "image",
    "images",
    "image_path",
    "image_paths",
    "frame",
    "frames",
    "frame_path",
    "frame_paths",
    "sampled_frames",
    "sampled_video_frame_paths",
]

VIDEO_KEYS = [
    "video",
    "videos",
    "video_path",
    "video_paths",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--path-mode", choices=["as-is", "absolute", "relative"], default="as-is")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--drop-missing-answer", action="store_true")
    parser.add_argument("--require-media-files", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def get_nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def first_text(row: dict[str, Any], paths: list[tuple[str, ...]]) -> str:
    for path in paths:
        text = text_value(get_nested(row, path))
        if text:
            return text
    return ""


def normalized_options(options: Any) -> str:
    if isinstance(options, dict):
        parts = []
        for key in sorted(options):
            value = text_value(options[key])
            if value:
                parts.append(f"{key}. {value}")
        return "\n".join(parts)
    if isinstance(options, list):
        parts = []
        for index, value in enumerate(options):
            text = text_value(value)
            if text:
                parts.append(f"{chr(ord('A') + index)}. {text}")
        return "\n".join(parts)
    return ""


def iter_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
    elif isinstance(value, dict):
        for key in ["path", "url", "image", "video", "file", "filename"]:
            if key in value:
                out.extend(iter_strings(value[key]))
        for key in ["images", "videos", "frames", "paths"]:
            if key in value:
                out.extend(iter_strings(value[key]))
    elif isinstance(value, list):
        for item in value:
            out.extend(iter_strings(item))
    return out


def collect_from_keys(row: dict[str, Any], keys: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    containers = [row]
    for key in ["meta", "metadata", "media"]:
        value = row.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in keys:
            for text in iter_strings(container.get(key)):
                if text not in seen:
                    out.append(text)
                    seen.add(text)
    return out


def collect_media(row: dict[str, Any], keys: list[str], suffixes: set[str]) -> list[str]:
    out = collect_from_keys(row, keys)
    for text in iter_strings(row.get("media")) + iter_strings(row.get("media_paths")):
        suffix = Path(text.split("?", 1)[0]).suffix.lower()
        if suffix in suffixes and text not in out:
            out.append(text)
    return out


def normalize_media_path(value: str, media_root: Path | None, path_mode: str) -> str:
    if URL_RE.match(value) or value.startswith("data:"):
        return value
    path = Path(value).expanduser()
    if media_root is not None and not path.is_absolute():
        path = media_root / path
    if path_mode == "absolute":
        return str(path.resolve(strict=False))
    if path_mode == "relative" and media_root is not None:
        try:
            return str(path.resolve(strict=False).relative_to(media_root.resolve(strict=False)))
        except ValueError:
            return str(path)
    return str(path)


def normalize_media(values: list[str], media_root: Path | None, path_mode: str, require_files: bool) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_media_path(value, media_root, path_mode)
        if require_files and not (URL_RE.match(text) or text.startswith("data:")):
            if not Path(text).expanduser().is_file():
                continue
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def tag_count(content: str, tag: str) -> int:
    return content.count(tag)


def add_missing_media_tags(content: str, images: list[str], videos: list[str]) -> str:
    prefix = ""
    image_missing = max(0, len(images) - tag_count(content, "<image>"))
    video_missing = max(0, len(videos) - tag_count(content, "<video>"))
    if image_missing:
        prefix += "<image>" * image_missing
    if video_missing:
        prefix += "<video>" * video_missing
    if prefix:
        return f"{prefix}\n{content}".strip()
    return content


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = text_value(message.get("role") or message.get("from")).lower()
        if role == "human":
            role = "user"
        elif role in {"gpt", "model"}:
            role = "assistant"
        content = message.get("content")
        if content is None:
            content = message.get("value")
        if role not in {"system", "user", "assistant", "tool", "tool_call", "tool_response"}:
            continue
        if isinstance(content, str):
            content_value: Any = content.strip()
        else:
            content_value = content
        if content_value in {"", None}:
            continue
        out.append({"role": role, "content": content_value})
    return out


def add_system_prompt(messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    text = system_prompt.strip()
    if not text or any(message.get("role") == "system" for message in messages):
        return messages
    return [{"role": "system", "content": text}] + messages


def add_media_tags_to_messages(messages: list[dict[str, Any]], images: list[str], videos: list[str]) -> list[dict[str, Any]]:
    if not images and not videos:
        return messages
    image_count = 0
    video_count = 0
    first_user_index: int | None = None
    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") == "user" and first_user_index is None and isinstance(content, str):
            first_user_index = index
        if isinstance(content, str):
            image_count += tag_count(content, "<image>")
            video_count += tag_count(content, "<video>")
    image_missing = max(0, len(images) - image_count)
    video_missing = max(0, len(videos) - video_count)
    if first_user_index is None or (image_missing == 0 and video_missing == 0):
        return messages
    prefix = "<image>" * image_missing + "<video>" * video_missing
    out = [dict(message) for message in messages]
    out[first_user_index]["content"] = f"{prefix}\n{out[first_user_index]['content']}".strip()
    return out


def build_messages(row: dict[str, Any], system_prompt: str, images: list[str], videos: list[str]) -> list[dict[str, Any]]:
    existing = normalize_messages(row.get("messages") or row.get("conversations") or row.get("conversation"))
    if existing:
        return add_media_tags_to_messages(add_system_prompt(existing, system_prompt), images, videos)
    question = first_text(row, QUESTION_PATHS)
    answer = first_text(row, ANSWER_PATHS)
    options = normalized_options(row.get("options") or row.get("choices"))
    if options:
        question = f"{question}\n\nOptions:\n{options}".strip()
    question = add_missing_media_tags(question, images, videos)
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    if question:
        messages.append({"role": "user", "content": question})
    if answer:
        messages.append({"role": "assistant", "content": answer})
    return messages


def is_valid_sft_sample(sample: dict[str, Any], drop_missing_answer: bool) -> bool:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return False
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
        return False
    if drop_missing_answer and not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        return False
    return True


def convert_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    images = normalize_media(
        collect_media(row, IMAGE_KEYS, IMAGE_SUFFIXES),
        args.media_root,
        args.path_mode,
        args.require_media_files,
    )
    videos = normalize_media(
        collect_media(row, VIDEO_KEYS, VIDEO_SUFFIXES),
        args.media_root,
        args.path_mode,
        args.require_media_files,
    )
    sample: dict[str, Any] = {"messages": build_messages(row, args.system_prompt, images, videos)}
    if images:
        sample["images"] = images
    if videos:
        sample["videos"] = videos
    return sample


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    samples = [convert_row(row, args) for row in rows]
    samples = [sample for sample in samples if is_valid_sft_sample(sample, args.drop_missing_answer)]
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(samples)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
