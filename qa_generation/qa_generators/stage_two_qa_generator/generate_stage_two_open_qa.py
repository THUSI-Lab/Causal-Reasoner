


from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_QA_ROOT = Path(os.environ.get("QA_ROOT", "qa_dataset"))
DEFAULT_TASKS = [
    "Task_13_Strategic_Rationale",
    "Task_14_Inter_Step_Dependency",
    "Task_18_Bad_Plan_Diagnosis_And_Repair",
    "Task_19_Counterfactual_Outcome",
    "Task_20_Failure_Recovery",
]

SYSTEM_PROMPT = """You are a precise embodied-AI QA answerer.
Your job is to answer the user's question directly from the provided evidence.

Rules:
- Use the provided images as the primary evidence whenever images are present.
- If auxiliary structured context is provided, use it only as supporting context.
- Do not mention filenames, paths, frame indices, timestamps, or source annotations.
- Do not describe your reasoning process.
- Return only the final answer text, with no markdown and no bullet list unless the task format explicitly requires it.
"""

TASK_GUIDANCE = {
    "Task_13_Strategic_Rationale": (
        "Answer the strategic-rationale question. "
        "Explain why the current step matters for achieving the high-level goal. "
        "Focus on plan-level necessity, not generic usefulness."
    ),
    "Task_14_Inter_Step_Dependency": (
        "Answer the inter-step dependency question. "
        "Explain how a real result of the previous step satisfies a key precondition for the next step. "
        "Focus on effect-to-precondition linkage, not weak workflow continuity."
    ),
    "Task_18_Bad_Plan_Diagnosis_And_Repair": (
        "Answer the bad-plan diagnosis-and-repair question. "
        "Output the result in the exact structured format requested in the question. "
        "Correctly identify the flaw step, flaw type, a short reason, and a repair plan."
    ),
    "Task_19_Counterfactual_Outcome": (
        "Answer the counterfactual-outcome question. "
        "Predict the single most likely immediate outcome under the stated condition. "
        "Do not propose any recovery action."
    ),
    "Task_20_Failure_Recovery": (
        "Answer the failure-recovery question. "
        "Give a concrete recovery action that directly addresses the stated failure and briefly explain why it works. "
        "Keep the recovery safe, hygienic, and practical."
    ),
}


@dataclass
class QaSample:
    task_name: str
    sample_id: str
    sample_index: int
    question: str
    reference_answer: str
    meta: dict[str, Any]
    raw_item: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open QA generation through an HTTP model service"
    )
    parser.add_argument("--qa-root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--http-base-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: generated_open_qa beside qa-root parent",
    )
    parser.add_argument(
        "--media-mode",
        choices=["stage3_keyframes", "video_uniform", "video_keyframe_biased"],
        default="video_keyframe_biased",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--one-step-images", type=int, default=32)
    parser.add_argument("--two-step-images-per-step", type=int, default=25)
    parser.add_argument("--prefix-images-per-step", type=int, default=6)
    parser.add_argument("--include-json-fallback", action="store_true", default=True)
    parser.add_argument("--json-fallback-mode", choices=["compact", "raw"], default="compact")
    parser.add_argument("--json-max-chars", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def conversations_to_question_answer(item: dict[str, Any]) -> tuple[str, str]:
    convs = item.get("conversations", [])
    question = convs[0].get("value", "") if len(convs) > 0 else ""
    answer = convs[1].get("value", "") if len(convs) > 1 else ""
    return question, answer


def load_qa_samples(task_dir: Path, limit: int) -> list[QaSample]:
    data_file = task_dir / "data.jsonl"
    samples: list[QaSample] = []
    with data_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            question, answer = conversations_to_question_answer(item)
            meta = item.get("meta", {})
            samples.append(
                QaSample(
                    task_name=meta.get("task_name", task_dir.name),
                    sample_id=item.get("id", ""),
                    sample_index=idx,
                    question=question,
                    reference_answer=answer,
                    meta=meta,
                    raw_item=item,
                )
            )
    return samples


def _parse_step_id_from_video_name(path_str: str) -> int | None:
    name = Path(path_str).name
    m = re.search(r"step(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"segment_start_to_step(\d+)_last", name)
    if m:
        return int(m.group(1))
    return None


def _step_ids_for_sample(sample: QaSample) -> list[int]:
    video = sample.raw_item.get("video")
    ids: list[int] = []
    if isinstance(video, list):
        for path_str in video:
            if isinstance(path_str, str):
                sid = _parse_step_id_from_video_name(path_str)
                if sid is not None:
                    ids.append(sid)
    elif isinstance(video, str):
        sid = _parse_step_id_from_video_name(video)
        if sid is not None:
            if sample.task_name == "Task_18_Bad_Plan_Diagnosis_And_Repair":
                ids.extend(list(range(1, sid + 1)))
            else:
                ids.append(sid)
    return ids


def _stage3_images_for_step(item_dir: str, step_id: int) -> list[str]:
    stage3_dir = Path(item_dir) / "stage3"
    prefix = f"{step_id:02d}_"
    if not stage3_dir.exists():
        return []
    matches = sorted([p for p in stage3_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not matches:
        return []
    imgs: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(sorted(str(p) for p in matches[0].glob(ext)))
    return imgs


def _extract_stage3_keyframe_indices(item_dir: str, step_id: int) -> list[int]:
    indices: list[int] = []
    for img in _stage3_images_for_step(item_dir, step_id):
        m = re.search(r"frame_(\d+)", Path(img).name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(set(indices))


def _sample_video_frames(video_path: str, max_images: int, critical_indices: list[int]) -> list[str]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to sample video frames.")
    path = Path(video_path)
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    selected: list[int] = []
    seen: set[int] = set()

    def add_idx(idx: int):
        if 0 <= idx < total and idx not in seen:
            selected.append(idx)
            seen.add(idx)

    if critical_indices:
        for idx in critical_indices:
            add_idx(int(idx) - 1)
        for off in (-8, -6, -4, -2, -1, 1, 2, 4, 6, 8):
            for idx in critical_indices:
                add_idx(int(idx) - 1 + off)

    while len(selected) < max_images:
        if max_images <= 1:
            add_idx(0)
            break
        uniform = [int(round(i * (total - 1) / max(1, max_images - 1))) for i in range(max_images)]
        changed = False
        for u in uniform:
            before = len(selected)
            add_idx(u)
            if len(selected) > before:
                changed = True
            if len(selected) >= max_images:
                break
        if not changed:
            break

    selected = selected[:max_images]
    out_dir = Path("/tmp/openqa_remote_frames") / re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    for idx, pos in enumerate(selected):
        out = out_dir / f"frame_{idx:03d}.jpg"
        if not out.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out), frame)
        out_paths.append(str(out))
    cap.release()
    return out_paths


def build_image_paths(sample: QaSample, args: argparse.Namespace) -> list[str]:
    item_dir = sample.meta.get("item_dir", "")
    if not item_dir:
        return []
    step_ids = _step_ids_for_sample(sample)
    if not step_ids:
        return []

    if args.media_mode == "stage3_keyframes":
        imgs: list[str] = []
        for sid in step_ids:
            imgs.extend(_stage3_images_for_step(item_dir, sid))
        return imgs[: args.max_images]

    video = sample.raw_item.get("video")
    video_paths: list[str] = video if isinstance(video, list) else ([video] if isinstance(video, str) else [])
    dense_images: list[str] = []

    if args.media_mode == "video_uniform":
        if len(video_paths) <= 1:
            for vp in video_paths:
                dense_images.extend(_sample_video_frames(vp, min(args.one_step_images, args.max_images), []))
        else:
            per_step = min(args.two_step_images_per_step, max(1, args.max_images // len(video_paths)))
            for vp in video_paths:
                dense_images.extend(_sample_video_frames(vp, per_step, []))
    elif args.media_mode == "video_keyframe_biased":
        if len(video_paths) <= 1:
            critical: list[int] = []
            for sid in step_ids:
                critical.extend(_extract_stage3_keyframe_indices(item_dir, sid))
            critical = sorted(set(critical))
            for vp in video_paths:
                dense_images.extend(
                    _sample_video_frames(vp, min(args.one_step_images, args.max_images), critical)
                )
        else:
            per_step = min(args.two_step_images_per_step, max(1, args.max_images // len(video_paths)))
            for sid, vp in zip(step_ids, video_paths):
                critical = _extract_stage3_keyframe_indices(item_dir, sid)
                dense_images.extend(_sample_video_frames(vp, per_step, critical))

    if dense_images:
        return dense_images[: args.max_images]

    imgs: list[str] = []
    for sid in step_ids:
        imgs.extend(_stage3_images_for_step(item_dir, sid))
    return imgs[: args.max_images]


def _compact_json_context(sample: QaSample) -> str:
    meta = sample.meta
    blocks = [f"task_name: {sample.task_name}"]
    if meta.get("source_path"):
        blocks.append(f"source_path: {meta['source_path']}")
    if meta.get("item_dir"):
        blocks.append(f"item_dir: {meta['item_dir']}")
    if meta.get("evidence_type"):
        blocks.append(f"evidence_type: {meta['evidence_type']}")
    if meta.get("evidence_files"):
        blocks.append("evidence_files: " + json.dumps(meta["evidence_files"], ensure_ascii=False))
    return "\n".join(blocks)


def _raw_json_fallback(sample: QaSample, args: argparse.Namespace) -> str:
    source_path = sample.meta.get("source_path", "")
    if not source_path:
        return ""
    text = Path(source_path).read_text(encoding="utf-8")
    if len(text) > args.json_max_chars:
        text = text[: args.json_max_chars] + "\n... [truncated]"
    return text


def build_user_text(sample: QaSample, args: argparse.Namespace, image_count: int) -> str:
    blocks = [
        f"Task: {sample.task_name}",
        f"Task-specific answering guidance:\n{TASK_GUIDANCE.get(sample.task_name, 'Answer the question directly and faithfully.')}",
        f"Question:\n{sample.question}",
    ]
    if args.include_json_fallback and image_count == 0:
        if args.json_fallback_mode == "raw":
            try:
                raw = _raw_json_fallback(sample, args)
            except Exception:
                raw = ""
            if raw:
                blocks.append(f"Auxiliary source JSON context:\n{raw}")
        else:
            blocks.append(f"Auxiliary source metadata:\n{_compact_json_context(sample)}")
    blocks.append("Return only the final answer text.")
    return "\n\n".join(blocks)


def _encode_image_path_to_data_url(path_str: str) -> str:
    path = Path(path_str)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_messages(sample: QaSample, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    image_paths = build_image_paths(sample, args)
    content: list[dict[str, Any]] = []
    for img in image_paths[: args.max_images]:
        content.append({"type": "input_image", "image_url": _encode_image_path_to_data_url(img)})
    content.append({"type": "input_text", "text": build_user_text(sample, args, len(image_paths))})
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, image_paths[: args.max_images]


def _call_worker_sync(
    base_url: str,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    reasoning_effort: str | None,
) -> str:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
    }
    if reasoning_effort and not model.lower().startswith("gpt-4o"):
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/invoke",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("status") != "ok":
        raise RuntimeError(data.get("error", "unknown worker error"))
    return data["result"]["output_text"]


async def generate_one(
    sample: QaSample,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages, used_images = build_messages(sample, args)
    text = await asyncio.to_thread(
        _call_worker_sync,
        args.http_base_url,
        messages,
        args.model,
        args.temperature,
        args.reasoning_effort,
    )
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "sample_index": sample.sample_index,
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "candidate_answer": text.strip(),
        "used_images": used_images,
        "image_count": len(used_images),
        "media_mode": args.media_mode,
        "meta": sample.meta,
    }


async def generate_one_with_semaphore(
    sample: QaSample,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        return await generate_one(sample, args)


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            existing[item["sample_id"]] = item
    return existing


async def main() -> None:
    args = parse_args()
    out_root = args.output_dir or (args.qa_root.parent / "generated_open_qa")
    out_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, args.workers))

    for task_name in args.tasks:
        task_dir = args.qa_root / task_name
        if not task_dir.exists():
            print(f"[WARN] skip missing task dir: {task_dir}")
            continue
        samples = load_qa_samples(task_dir, args.limit)
        task_out_dir = out_root / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = task_out_dir / "predictions.jsonl"
        summary_path = task_out_dir / "summary.json"

        existing = load_existing_results(jsonl_path) if args.resume else {}
        processed_ids = set(existing.keys())
        pending = [s for s in samples if s.sample_id not in processed_ids]
        rows = list(existing.values())
        tasks = [asyncio.create_task(generate_one_with_semaphore(s, args, semaphore)) for s in pending]

        write_mode = "a" if args.resume and jsonl_path.exists() else "w"
        with jsonl_path.open(write_mode, encoding="utf-8") as fout:
            completed = 0
            total = len(pending)
            for fut in asyncio.as_completed(tasks):
                row = await fut
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                rows.append(row)
                completed += 1
                print(f"[{task_name}] {completed}/{total} sample={row['sample_id']} images={row['image_count']}")

        summary = {
            "task_name": task_name,
            "total": len(rows),
            "media_mode": args.media_mode,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "avg_image_count": round(sum(r.get("image_count", 0) for r in rows) / max(1, len(rows)), 4),
            "qa_root": str(args.qa_root),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] output_dir={out_root}")


if __name__ == "__main__":
    asyncio.run(main())
