from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

import open_qa_judge_rubric_prompts_en


BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE_DIR.parent
DEFAULT_BENCHMARK_DATA_ROOT = Path(
    os.environ.get("BENCHMARK_DATA_ROOT", REPOSITORY_ROOT / "benchmark_data")
).expanduser()
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("EVALUATION_OUTPUT_ROOT", REPOSITORY_ROOT / "eval_outputs")).expanduser()
DEFAULT_MEDIA_CACHE_DIR = Path(os.environ.get("EVALUATION_MEDIA_CACHE_DIR", REPOSITORY_ROOT / ".eval_media_cache")).expanduser()
DEFAULT_REGISTRY_PATH = BASE_DIR / "open_qa_model_registry.json"
ACTIVE_BENCHMARK_DATA_ROOT = DEFAULT_BENCHMARK_DATA_ROOT

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_PROFILE = os.environ.get("AZURE_OPENAI_API_PROFILE") or os.environ.get("API_PROFILE", "")
AZURE_CONFIG_DIR = Path(os.environ.get("AZURE_CONFIG_DIR", "/tmp/azure_cli_config")).expanduser()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PRO_IMAGE_LIMIT = 50
DEFAULT_PRO_REASONING_EFFORT = "medium"


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    model: str
    reasoning_effort: str | None
    max_output_tokens: int


@dataclass(frozen=True)
class BenchmarkItem:
    split: str
    task_name: str
    sample_id: str
    sample_index: int
    source_file: Path
    raw: dict[str, Any]
    question: str
    reference_answer: str
    options: dict[str, str] | None = None
    gold_letter: str | None = None


@dataclass(frozen=True)
class MediaSelection:
    evidence_type: str
    image_paths: list[str]
    video_paths: list[str]
    attached_image_paths: list[str]
    sampled_video_frame_paths: list[str]
    sampled_video_frame_groups: list[dict[str, Any]]


_AZURE_RESPONSES_CLIENT: AzureOpenAI | None = None


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "run"


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prompt_module() -> dict[str, Any]:
    py_prompts = open_qa_judge_rubric_prompts_en.TASK_PROMPTS
    py_titles = open_qa_judge_rubric_prompts_en.TASK_TITLES
    if set(py_titles) != set(py_prompts):
        raise SystemExit(
            "open_qa_judge_rubric_prompts_en.py has inconsistent TASK_TITLES/TASK_PROMPTS keys: "
            f"title_keys={sorted(py_titles)} prompt_keys={sorted(py_prompts)}"
        )
    empty_keys = [key for key, value in py_prompts.items() if not str(value or "").strip()]
    if empty_keys:
        raise SystemExit(f"Empty prompt text in prompt module for: {', '.join(empty_keys)}")
    han_re = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    chinese_keys = [key for key, value in py_prompts.items() if han_re.search(value)]
    if chinese_keys:
        raise SystemExit(f"Chinese text remains in prompt module for: {', '.join(chinese_keys)}")
    prompt_hashes = {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(py_prompts.items())
    }
    return {
        "prompt_module": str(Path(open_qa_judge_rubric_prompts_en.__file__).resolve()),
        "prompt_count": len(py_prompts),
        "prompt_hashes_sha256": prompt_hashes,
        "module_check_passed": True,
    }


def normalize_reasoning_effort(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null"}:
        return None
    return text


def resolve_model_config(
    alias_or_model: str,
    registry: dict[str, Any],
    *,
    reasoning_effort_override: str | None,
    max_output_tokens_override: int | None,
) -> ModelConfig:
    models = registry.get("models") if isinstance(registry, dict) else {}
    if not isinstance(models, dict):
        models = {}
    raw = models.get(alias_or_model)
    if raw is None:
        raw = {
            "model": alias_or_model,
            "reasoning_effort": None,
            "max_output_tokens": 900,
        }

    model = str(raw.get("model") or alias_or_model).strip()
    if not model:
        raise SystemExit(f"Empty model name for {alias_or_model!r}")

    reasoning_effort = normalize_reasoning_effort(
        reasoning_effort_override if reasoning_effort_override is not None else raw.get("reasoning_effort")
    )
    max_output_tokens = int(
        max_output_tokens_override
        if max_output_tokens_override is not None
        else raw.get("max_output_tokens", 900)
    )
    if max_output_tokens <= 0:
        raise SystemExit(f"Invalid max_output_tokens for {alias_or_model!r}: {max_output_tokens}")

    return ModelConfig(
        alias=alias_or_model,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )


def init_azure_responses_client() -> AzureOpenAI:
    global _AZURE_RESPONSES_CLIENT
    if _AZURE_RESPONSES_CLIENT is None:
        if not AZURE_OPENAI_ENDPOINT:
            raise SystemExit("Set AZURE_OPENAI_ENDPOINT before running model-backed evaluation.")
        if not AZURE_OPENAI_API_PROFILE:
            raise SystemExit("Set AZURE_OPENAI_API_PROFILE or API_PROFILE before running Azure-backed evaluation.")
        source_azure_dir = Path.home() / ".azure"
        AZURE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if source_azure_dir.is_dir():
            for src in source_azure_dir.iterdir():
                dst = AZURE_CONFIG_DIR / src.name
                try:
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    elif src.is_file():
                        shutil.copy2(src, dst)
                except Exception:
                    pass
        os.environ.setdefault("AZURE_CONFIG_DIR", str(AZURE_CONFIG_DIR))
        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        client_kwargs: dict[str, Any] = {
            "azure_endpoint": AZURE_OPENAI_ENDPOINT,
            "azure_ad_token_provider": token_provider,
        }
        client_kwargs["api_" + "ver" + "sion"] = AZURE_OPENAI_API_PROFILE
        _AZURE_RESPONSES_CLIENT = AzureOpenAI(**client_kwargs)
        print("[azure] endpoint=<configured>")
    return _AZURE_RESPONSES_CLIENT


def resolve_responses_reasoning(model: str, effort: str | None) -> dict[str, str] | None:
    model_name = model.strip().lower()
    value = normalize_reasoning_effort(effort)
    if model_name == "gpt-5.4-pro":
        return {"effort": value or DEFAULT_PRO_REASONING_EFFORT}
    if value is None:
        return None
    return {"effort": value}


def extract_responses_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            ctext = getattr(content, "text", None)
            if ctext:
                parts.append(ctext)
    return "\n".join(parts).strip()


def encode_image_path_to_data_url(path_str: str) -> str:
    path = Path(path_str).expanduser()
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def materialize_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in content:
        item_type = item.get("type")
        if item_type in {"input_text", "text"}:
            out.append({"type": "input_text", "text": str(item.get("text", ""))})
        elif item_type == "input_image_path":
            path_str = str(item.get("path") or "")
            if not path_str:
                raise ValueError("input_image_path missing path")
            out.append({"type": "input_image", "image_url": encode_image_path_to_data_url(path_str)})
        elif item_type == "input_image":
            image_url = item.get("image_url")
            if isinstance(image_url, str) and image_url and not image_url.startswith("data:") and Path(image_url).exists():
                out.append({"type": "input_image", "image_url": encode_image_path_to_data_url(image_url)})
            else:
                out.append(item)
        else:
            out.append(item)
    return out


def materialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    realized: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = materialize_content(content)
        elif isinstance(content, str):
            msg["content"] = [{"type": "input_text", "text": content}]
        realized.append(msg)
    return realized


def count_input_images(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content") or []
        if isinstance(content, list):
            count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "input_image")
    return count


def call_azure_responses_sync(
    *,
    messages: list[dict[str, Any]],
    model_cfg: ModelConfig,
    timeout: float,
    max_output_tokens: int | None = None,
) -> str:
    client = init_azure_responses_client()
    payload_messages = materialize_messages(messages)
    image_count = count_input_images(payload_messages)
    if image_count > PRO_IMAGE_LIMIT:
        raise ValueError(f"request has {image_count} images; hard limit is {PRO_IMAGE_LIMIT}")

    kwargs: dict[str, Any] = {
        "model": model_cfg.model,
        "input": payload_messages,
        "max_output_tokens": int(max_output_tokens or model_cfg.max_output_tokens),
    }
    reasoning = resolve_responses_reasoning(model_cfg.model, model_cfg.reasoning_effort)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    resp = client.with_options(timeout=float(timeout)).responses.create(**kwargs)
    text = extract_responses_text(resp)
    if text:
        return text

    output_items = getattr(resp, "output", []) or []
    if any(getattr(item, "type", None) == "reasoning" for item in output_items):
        retry_kwargs = dict(kwargs)
        retry_kwargs["reasoning"] = {"effort": "medium"}
        retry_kwargs["max_output_tokens"] = max(1200, int(max_output_tokens or model_cfg.max_output_tokens))
        retry_resp = client.with_options(timeout=float(timeout)).responses.create(**retry_kwargs)
        return extract_responses_text(retry_resp)
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def conversations_to_question_answer(item: dict[str, Any]) -> tuple[str, str]:
    convs = item.get("conversations") or []
    question = ""
    answer = ""
    if len(convs) >= 1 and isinstance(convs[0], dict):
        question = str(convs[0].get("value") or "").strip()
    if len(convs) >= 2 and isinstance(convs[1], dict):
        answer = str(convs[1].get("value") or "").strip()
    return question, answer


def load_items(benchmark_data_root: Path, split: str, tasks: list[str] | None) -> list[BenchmarkItem]:
    global ACTIVE_BENCHMARK_DATA_ROOT
    ACTIVE_BENCHMARK_DATA_ROOT = benchmark_data_root.expanduser().resolve()
    split_root = benchmark_data_root / split
    if not split_root.is_dir():
        raise SystemExit(f"Missing split directory: {split_root}")

    selected_tasks = set(tasks or [])
    items: list[BenchmarkItem] = []
    for data_path in sorted(split_root.glob("*/data.jsonl")):
        task_name = data_path.parent.name
        if selected_tasks and task_name not in selected_tasks:
            continue
        for idx, raw in enumerate(read_jsonl(data_path)):
            sample_id = str(raw.get("sample_id") or raw.get("id") or "").strip()
            if not sample_id:
                raise ValueError(f"Missing sample id in {data_path}:{idx + 1}")
            if split == "mcq":
                options = raw.get("options")
                if not isinstance(options, dict):
                    raise ValueError(f"MCQ row missing options: {data_path}:{idx + 1}")
                question = str(raw.get("question") or "").strip()
                gold_letter = str(raw.get("answer") or "").strip().upper()
                reference_answer = str(raw.get("answer_text") or raw.get("gold_answer") or "").strip()
                if not question or gold_letter not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Invalid MCQ row: {data_path}:{idx + 1}")
                items.append(
                    BenchmarkItem(
                        split=split,
                        task_name=task_name,
                        sample_id=sample_id,
                        sample_index=idx,
                        source_file=data_path,
                        raw=raw,
                        question=question,
                        reference_answer=reference_answer,
                        options={k: str(v) for k, v in options.items()},
                        gold_letter=gold_letter,
                    )
                )
            else:
                question = str(raw.get("question") or raw.get("prompt") or "").strip()
                reference_answer = str(raw.get("reference_answer") or raw.get("answer") or "").strip()
                if not question or not reference_answer:
                    q_conv, a_conv = conversations_to_question_answer(raw)
                    question = question or q_conv
                    reference_answer = reference_answer or a_conv
                if not question or not reference_answer:
                    raise ValueError(f"Invalid QA row: {data_path}:{idx + 1}")
                items.append(
                    BenchmarkItem(
                        split=split,
                        task_name=task_name,
                        sample_id=sample_id,
                        sample_index=idx,
                        source_file=data_path,
                        raw=raw,
                        question=question,
                        reference_answer=reference_answer,
                    )
                )
    return items


def task_selected_items(
    items: list[BenchmarkItem],
    *,
    limit_per_task: int,
    max_items_total: int,
    smoke_one: bool = False,
) -> list[BenchmarkItem]:
    if smoke_one:
        video_items = [item for item in items if collect_existing_media_paths(item)[1]]
        return (video_items or items)[:1]

    by_task: dict[str, list[BenchmarkItem]] = {}
    for item in items:
        by_task.setdefault(item.task_name, []).append(item)

    selected: list[BenchmarkItem] = []
    for task_name in sorted(by_task):
        task_items = by_task[task_name]
        if limit_per_task > 0:
            task_items = task_items[:limit_per_task]
        selected.extend(task_items)

    if max_items_total > 0:
        selected = selected[:max_items_total]
    return selected


def raw_path_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in seen:
                out.append(text)
                seen.add(text)
        elif isinstance(value, list):
            for sub_value in value:
                if isinstance(sub_value, str):
                    text = sub_value.strip()
                    if text and text not in seen:
                        out.append(text)
                        seen.add(text)
    return out


def normalize_existing_paths(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_path_list(values):
        raw_path = Path(value).expanduser()
        candidates = [raw_path] if raw_path.is_absolute() else [
            ACTIVE_BENCHMARK_DATA_ROOT / raw_path,
            DEFAULT_BENCHMARK_DATA_ROOT / raw_path,
            REPOSITORY_ROOT / raw_path,
            Path.cwd() / raw_path,
        ]
        for candidate in candidates:
            try:
                path = candidate.resolve()
                if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            text = str(path)
            if text not in seen:
                out.append(text)
                seen.add(text)
            break
    return out


def fallback_media_under_item_dir(item: BenchmarkItem) -> list[str]:
    raw = item.raw
    item_dir = raw.get("item_dir") or (raw.get("meta") or {}).get("item_dir")
    if not isinstance(item_dir, str) or not item_dir.strip():
        return []
    raw_root = Path(item_dir).expanduser()
    root_candidates = [raw_root] if raw_root.is_absolute() else [
        ACTIVE_BENCHMARK_DATA_ROOT / raw_root,
        DEFAULT_BENCHMARK_DATA_ROOT / raw_root,
        REPOSITORY_ROOT / raw_root,
        Path.cwd() / raw_root,
    ]
    root = next((candidate.resolve() for candidate in root_candidates if candidate.is_dir()), raw_root)
    if not root.is_dir():
        return []
    media_dir = root / "media"
    search_root = media_dir if media_dir.is_dir() else root
    paths = [
        str(p)
        for p in sorted(search_root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES and p.stat().st_size > 0
    ]
    return paths


def collect_existing_media_paths(item: BenchmarkItem) -> tuple[list[str], list[str]]:
    raw = item.raw
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    image_values: list[Any] = []
    video_values: list[Any] = []
    fallback_values: list[Any] = []

    if item.split == "mcq":
        image_values.extend([raw.get("source_image_paths")])
        video_values.extend([raw.get("source_video_paths")])
        fallback_values.extend([raw.get("source_multimodal_paths"), raw.get("source_evidence_paths")])
    else:
        image_values.extend([raw.get("image"), meta.get("source_image_paths")])
        video_values.extend([raw.get("video"), meta.get("source_video_paths")])
        fallback_values.extend([meta.get("evidence_files"), raw.get("source_multimodal_paths"), meta.get("source_multimodal_paths")])

    image_paths: list[str] = []
    video_paths: list[str] = []
    for values in image_values:
        image_paths.extend(normalize_existing_paths(values))
    for values in video_values:
        video_paths.extend(normalize_existing_paths(values))

    if not image_paths and not video_paths:
        fallback_paths: list[str] = []
        for values in fallback_values:
            fallback_paths.extend(normalize_existing_paths(values))
        if not fallback_paths:
            fallback_paths = normalize_existing_paths(fallback_media_under_item_dir(item))
        for path_str in fallback_paths:
            suffix = Path(path_str).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                image_paths.append(path_str)
            elif suffix in VIDEO_SUFFIXES:
                video_paths.append(path_str)

    return dedupe(image_paths), dedupe(video_paths)


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def ffprobe_duration_seconds(video_path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = proc.stdout.strip()
        return float(raw) if raw else None
    except Exception:
        return None


def video_frame_cache_dir(media_cache_dir: Path, video_path: Path, max_frames: int) -> Path:
    stat = video_path.stat()
    key = hashlib.sha1(f"{video_path}:{stat.st_size}:{stat.st_mtime_ns}:{max_frames}".encode("utf-8")).hexdigest()[:16]
    return media_cache_dir / key


def extract_video_frames(video_path_str: str, max_frames: int, media_cache_dir: Path) -> list[str]:
    if max_frames <= 0:
        return []
    video_path = Path(video_path_str)
    if not video_path.exists() or not video_path.is_file():
        return []
    out_dir = video_frame_cache_dir(media_cache_dir, video_path, max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(str(p) for p in out_dir.glob("frame_*.jpg"))
    if existing:
        return existing[:max_frames]

    duration = max(ffprobe_duration_seconds(video_path) or 4.0, 0.5)
    fps = max(max_frames / duration, 0.25)
    pattern = out_dir / "frame_%03d.jpg"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                "-frames:v",
                str(max_frames),
                str(pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return sorted(str(p) for p in out_dir.glob("frame_*.jpg"))[:max_frames]


def select_media(item: BenchmarkItem, args: argparse.Namespace) -> MediaSelection:
    image_paths, video_paths = collect_existing_media_paths(item)
    if not image_paths and not video_paths:
        raise ValueError(f"source_multimodal_missing:{item.split}:{item.task_name}:{item.sample_id}")

    attached_images = image_paths[: max(0, args.image_max_count)]
    sampled_frames: list[str] = []
    sampled_groups: list[dict[str, Any]] = []
    if video_paths:
        attached_images = []
        if len(video_paths) >= 2:
            for video_index, video_path in enumerate(video_paths[:2], 1):
                group_frames = extract_video_frames(video_path, max(0, args.video_clip_frames), args.media_cache_dir)
                sampled_groups.append(
                    {
                        "video_index": video_index,
                        "video_path": video_path,
                        "frame_paths": group_frames,
                    }
                )
                sampled_frames.extend(group_frames)
            frame_budget = max(0, args.video_max_frames)
            if frame_budget <= 0:
                sampled_frames = []
                sampled_groups = []
            elif len(sampled_frames) > frame_budget:
                remaining = frame_budget
                truncated_groups: list[dict[str, Any]] = []
                sampled_frames = []
                for group in sampled_groups:
                    keep = list(group.get("frame_paths") or [])[:remaining]
                    remaining -= len(keep)
                    sampled_frames.extend(keep)
                    truncated_group = dict(group)
                    truncated_group["frame_paths"] = keep
                    truncated_groups.append(truncated_group)
                    if remaining <= 0:
                        break
                sampled_groups = truncated_groups
        else:
            sampled_frames = extract_video_frames(video_paths[0], max(0, args.video_max_frames), args.media_cache_dir)
            sampled_groups.append(
                {
                    "video_index": 1,
                    "video_path": video_paths[0],
                    "frame_paths": sampled_frames,
                }
            )

    evidence_type = str(
        item.raw.get("source_evidence_type")
        or (item.raw.get("meta") or {}).get("evidence_type")
        or ("video" if video_paths else "image")
    )
    return MediaSelection(
        evidence_type=evidence_type,
        image_paths=image_paths,
        video_paths=video_paths,
        attached_image_paths=attached_images,
        sampled_video_frame_paths=sampled_frames,
        sampled_video_frame_groups=sampled_groups,
    )


def media_content_blocks(media: MediaSelection) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for image_index, path in enumerate(media.attached_image_paths, 1):
        blocks.append({"type": "input_text", "text": f"Evidence image {image_index}."})
        blocks.append({"type": "input_image_path", "path": path})
    for group in media.sampled_video_frame_groups:
        frame_paths = group.get("frame_paths") or []
        if not frame_paths:
            continue
        blocks.append(
            {
                "type": "input_text",
                "text": (
                    f"Video clip {group.get('video_index')} sampled frames follow in chronological order. "
                    "Treat these frames as visual evidence from that original clip."
                ),
            }
        )
        for frame_index, path in enumerate(frame_paths, 1):
            blocks.append({"type": "input_text", "text": f"Video clip {group.get('video_index')} frame {frame_index}."})
            blocks.append({"type": "input_image_path", "path": path})
    if not blocks:
        raise ValueError("no_attachable_media_after_selection")
    return blocks


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    if not cleaned:
        raise ValueError("empty_model_output")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def media_meta(media: MediaSelection) -> dict[str, Any]:
    return {
        "evidence_type": media.evidence_type,
        "source_image_paths": media.image_paths,
        "source_video_paths": media.video_paths,
        "attached_image_paths": media.attached_image_paths,
        "sampled_video_frame_paths": media.sampled_video_frame_paths,
        "sampled_video_frame_groups": media.sampled_video_frame_groups,
        "attached_image_count": len(media.attached_image_paths) + len(media.sampled_video_frame_paths),
    }


def existing_result_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                ids.add(sample_id)
    return ids


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid result JSON in {path}:{line_no}") from exc
    return rows


def duplicate_sample_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in seen and sample_id not in dupes:
            dupes.append(sample_id)
        seen.add(sample_id)
    return dupes


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
