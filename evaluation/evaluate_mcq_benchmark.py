from __future__ import annotations

import argparse
import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

import evaluation_common as common


MCQ_SYSTEM_PROMPT = (
    "You are a strict multimodal multiple-choice solver. Use only the provided "
    "multimodal evidence, question, and answer options. Output valid JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict MCQ evaluation for the 600-item MCQ split.")
    parser.add_argument("--benchmark-data-root", type=Path, default=common.DEFAULT_BENCHMARK_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=common.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional MCQ task folder names to run.")
    parser.add_argument("--model", default="gpt-5.4", help="Target model for MCQ solving.")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3000.0)
    parser.add_argument("--image-max-count", type=int, default=1)
    parser.add_argument("--video-max-frames", type=int, default=50)
    parser.add_argument("--video-clip-frames", type=int, default=25)
    parser.add_argument("--media-cache-dir", type=Path, default=common.DEFAULT_MEDIA_CACHE_DIR)
    parser.add_argument("--limit-per-task", type=int, default=0, help="0 means all items.")
    parser.add_argument("--max-items-total", type=int, default=0, help="Optional cap after task selection.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already in output JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output dir before running.")
    parser.add_argument("--dry-run", action="store_true", help="Validate loading/media only; do not call models.")
    return parser.parse_args()


def build_mcq_messages(item: common.BenchmarkItem, media: common.MediaSelection) -> list[dict[str, Any]]:
    assert item.options is not None
    options_text = "\n".join(f"{letter}: {item.options.get(letter, '')}" for letter in ["A", "B", "C", "D"])
    user_text = "\n\n".join(
        [
            "Use the attached multimodal evidence to answer this multiple-choice question.",
            f"Task: {item.task_name}",
            f"Question:\n{item.question}",
            f"Options:\n{options_text}",
            (
                "Return valid JSON only. The JSON object must have exactly these keys:\n"
                '- "answer": one of "A", "B", "C", or "D"\n'
                '- "reason": one short sentence explaining the choice'
            ),
        ]
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": MCQ_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Multimodal evidence for this exact MCQ item follows."},
                *common.media_content_blocks(media),
                {"type": "input_text", "text": user_text},
            ],
        },
    ]


def extract_mcq_letter(response: str) -> str | None:
    if not response:
        return None
    try:
        parsed = common.parse_model_json(response)
        value = str(parsed.get("answer") or "").strip().upper()
        if value in {"A", "B", "C", "D"}:
            return value
    except Exception:
        pass

    text = common.strip_code_fence(response)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    patterns = [
        r'"answer"\s*:\s*"([A-D])"',
        r"(?:final\s+answer|answer|choice)\s*[:]\s*(?:option\s*)?([A-D])\b",
        r"(?:choose|select|pick)\s+(?:option\s+)?([A-D])\b",
        r"\boption\s+([A-D])\b",
        r"^\s*([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    tail = text[-400:]
    matches = list(re.finditer(r"\b([A-D])\b", tail, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()
    return None


async def run_mcq_item(
    item: common.BenchmarkItem,
    model_cfg: common.ModelConfig,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        media = common.select_media(item, args)
        if args.dry_run:
            raw_response = ""
            predicted = None
        else:
            messages = build_mcq_messages(item, media)
            raw_response = await asyncio.to_thread(
                common.call_azure_responses_sync,
                messages=messages,
                model_cfg=model_cfg,
                timeout=args.timeout,
            )
            predicted = extract_mcq_letter(raw_response)
        correct = bool(predicted and predicted == item.gold_letter)
        return {
            "split": "mcq",
            "task_name": item.task_name,
            "sample_id": item.sample_id,
            "sample_index": item.sample_index,
            "source_file": str(item.source_file),
            "question": item.question,
            "options": item.options,
            "gold": item.gold_letter,
            "gold_answer_text": item.reference_answer,
            "predicted": predicted,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "model": model_cfg.model,
            "raw_model_output": raw_response,
            "media": common.media_meta(media),
        }


def summarize_mcq(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answered = sum(1 for row in rows if row.get("predicted") in {"A", "B", "C", "D"})
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "total": total,
        "answered": answered,
        "failed_to_parse": total - answered,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_on_answered": round(correct / answered, 4) if answered else 0.0,
    }


async def run_mcq_split(
    items: list[common.BenchmarkItem],
    model_cfg: common.ModelConfig,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    by_task: dict[str, list[common.BenchmarkItem]] = {}
    for item in items:
        by_task.setdefault(item.task_name, []).append(item)

    split_summary: dict[str, Any] = {}
    for task_name in sorted(by_task):
        task_items = by_task[task_name]
        task_dir = output_dir / "mcq" / task_name
        result_path = task_dir / "mcq_results.jsonl"
        if result_path.exists() and not args.resume and not args.overwrite:
            raise SystemExit(
                f"Refusing to append duplicate results to existing file without --resume or --overwrite: {result_path}"
            )
        if args.resume:
            done_ids = common.existing_result_ids(result_path)
            task_items = [item for item in task_items if item.sample_id not in done_ids]

        print(f"[mcq] {task_name}: running {len(task_items)} item(s)")
        if task_items:
            coros = [run_mcq_item(item, model_cfg, args, semaphore) for item in task_items]
            rows = [await coro for coro in asyncio.as_completed(coros)]
            rows.sort(key=lambda row: int(row.get("sample_index", 0)))
            common.append_jsonl(result_path, rows)

        all_rows = common.load_result_rows(result_path)
        if not all_rows:
            continue
        duplicates = common.duplicate_sample_ids(all_rows)
        if duplicates:
            raise SystemExit(f"Duplicate sample_id rows in {result_path}: {duplicates[:10]}")
        task_summary = summarize_mcq(all_rows)
        task_summary.update(
            {
                "split": "mcq",
                "task_name": task_name,
                "result_path": str(result_path),
                "model": model_cfg.model,
            }
        )
        common.write_json(task_dir / "summary.json", task_summary)
        split_summary[task_name] = task_summary
    return split_summary


def build_overall_summary(mcq_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    total = sum(v.get("total", 0) for v in mcq_summary.values())
    answered = sum(v.get("answered", 0) for v in mcq_summary.values())
    correct = sum(v.get("correct", 0) for v in mcq_summary.values())
    return {
        "output_dir": str(output_dir),
        "split": "mcq",
        "total": total,
        "answered": answered,
        "failed_to_parse": total - answered,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_on_answered": round(correct / answered, 4) if answered else 0.0,
        "by_task": mcq_summary,
    }


async def async_main() -> None:
    args = parse_args()
    args.benchmark_data_root = args.benchmark_data_root.expanduser().resolve()
    args.media_cache_dir = args.media_cache_dir.expanduser().resolve()
    args.media_cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.benchmark_data_root.exists():
        raise SystemExit(f"Benchmark data root not found: {args.benchmark_data_root}")

    registry = common.read_registry(args.registry.expanduser().resolve())
    model_cfg = common.resolve_model_config(
        args.model,
        registry,
        reasoning_effort_override=args.reasoning_effort,
        max_output_tokens_override=args.max_output_tokens,
    )

    if args.output_dir is None:
        output_dir = common.DEFAULT_OUTPUT_ROOT / f"mcq__{common.sanitize_name(model_cfg.alias)}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        common.init_azure_responses_client()

    manifest = {
        "benchmark_data_root": str(args.benchmark_data_root),
        "split": "mcq",
        "tasks": args.tasks,
        "target_model": model_cfg.__dict__,
        "image_max_count": args.image_max_count,
        "video_max_frames": args.video_max_frames,
        "video_clip_frames": args.video_clip_frames,
        "dry_run": args.dry_run,
    }
    common.write_json(output_dir / "manifest.json", manifest)

    all_items = common.load_items(args.benchmark_data_root, "mcq", args.tasks)
    items = common.task_selected_items(
        all_items,
        limit_per_task=args.limit_per_task,
        max_items_total=args.max_items_total,
    )
    print(f"[mcq] selected {len(items)} item(s) from {len(all_items)}")
    mcq_summary = await run_mcq_split(items, model_cfg, output_dir, args)
    overall = build_overall_summary(mcq_summary, output_dir)
    common.write_json(output_dir / "summary.json", overall)
    print(f"[done] MCQ summary: {output_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
