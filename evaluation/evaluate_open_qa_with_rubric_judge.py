from __future__ import annotations

import argparse
import asyncio
import math
import shutil
from pathlib import Path
from typing import Any

import evaluation_common as common
import open_qa_judge_rubric_prompts_en


SCORE_MIN = 0.0
SCORE_MAX = 1.0
SCORE_PRECISION = 3
SCORE_SCHEMA = "continuous_0_1_task_specific_rubric_from_open_qa_judge_rubric_prompts_en"

QA_GENERATION_SYSTEM_PROMPT = (
    "You are a precise embodied-AI QA answerer. Answer only from the provided "
    "multimodal evidence and the user question. Do not mention filenames, paths, "
    "frame indices, source annotations, or the reference answer. "
    "Return only the final answer text."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict JSON judge. Follow the provided task-specific rubric exactly. "
    "Output valid JSON only, with no markdown fences and no extra commentary."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict open-QA generation and rubric judging.")
    parser.add_argument("--benchmark-data-root", type=Path, default=common.DEFAULT_BENCHMARK_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=common.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional QA task folder names to run.")
    parser.add_argument("--model", default="gpt-5.4", help="Target model for candidate answer generation.")
    parser.add_argument("--judge-model", default="gpt-5.4", help="Rubric judge model for scoring.")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--judge-reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--judge-max-output-tokens", type=int, default=1200)
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
    parser.add_argument(
        "--judge-parse-error-retries",
        type=int,
        default=2,
        help="Retry rubric judging when the judge output is not valid score JSON.",
    )
    parser.add_argument(
        "--skip-prompt-module-check",
        "--skip-prompt-sync-check",
        dest="skip_prompt_module_check",
        action="store_true",
        help="Do not verify open_qa_judge_rubric_prompts_en.py internal consistency.",
    )
    return parser.parse_args()


def build_qa_generation_messages(item: common.BenchmarkItem, media: common.MediaSelection) -> list[dict[str, Any]]:
    user_text = "\n\n".join(
        [
            "Use the attached multimodal evidence to answer this open question.",
            f"Task: {item.task_name}",
            f"Question:\n{item.question}",
            "Return only the final answer text.",
        ]
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": QA_GENERATION_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Multimodal evidence for this exact QA item follows."},
                *common.media_content_blocks(media),
                {"type": "input_text", "text": user_text},
            ],
        },
    ]


def build_qa_judge_messages(
    item: common.BenchmarkItem,
    candidate_answer: str,
    media: common.MediaSelection,
) -> list[dict[str, Any]]:
    prompt_task_name = str((item.raw.get("meta") or {}).get("task_name") or item.task_name)
    task_prompt = open_qa_judge_rubric_prompts_en.get_prompt(prompt_task_name)
    user_tail = "\n\n".join(
        [
            f"Task name:\n{prompt_task_name}",
            f"Sample ID:\n{item.sample_id}",
            f"Question:\n{item.question}",
            f"Gold standard answer:\n{item.reference_answer}",
            f"Candidate answer:\n{candidate_answer}",
            "Return valid JSON only.",
        ]
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": JUDGE_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": task_prompt.strip()},
                {"type": "input_text", "text": "Multimodal evidence for this exact QA item follows."},
                *common.media_content_blocks(media),
                {"type": "input_text", "text": user_tail},
            ],
        },
    ]


def score_band_label(score: float) -> str:
    if score < 0.25:
        return "[0.000,0.250)"
    if score < 0.5:
        return "[0.250,0.500)"
    if score < 0.75:
        return "[0.500,0.750)"
    if score <= 1.0:
        return "[0.750,1.000]"
    raise ValueError(f"score_out_of_range:{score}")


def parse_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("invalid_score_type:bool")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        score = float(value.strip())
    else:
        raise ValueError("missing_score")
    if not math.isfinite(score):
        raise ValueError("non_finite_score")
    score = round(score, SCORE_PRECISION)
    if score < SCORE_MIN or score > SCORE_MAX:
        raise ValueError(f"invalid_score:{score}")
    return score


def parse_judge_output(raw: str) -> dict[str, Any]:
    parsed = common.parse_model_json(raw)
    score = parse_score(parsed.get("score"))
    reason = str(parsed.get("reason") or "").strip()
    return {
        "score": score,
        "score_band": score_band_label(score),
        "reason": reason,
        "score_schema": SCORE_SCHEMA,
        "score_precision": SCORE_PRECISION,
        "parse_error": False,
    }


async def run_qa_item(
    item: common.BenchmarkItem,
    answer_model_cfg: common.ModelConfig,
    judge_model_cfg: common.ModelConfig,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        media = common.select_media(item, args)
        if args.dry_run:
            candidate_answer = ""
            raw_answer_output = ""
            raw_judge_output = ""
            judge = {
                "score": -1.0,
                "score_band": "dry_run",
                "reason": "Dry run did not call the answer or judge model.",
                "score_schema": SCORE_SCHEMA,
                "parse_error": True,
            }
        else:
            answer_messages = build_qa_generation_messages(item, media)
            raw_answer_output = await asyncio.to_thread(
                common.call_azure_responses_sync,
                messages=answer_messages,
                model_cfg=answer_model_cfg,
                timeout=args.timeout,
            )
            candidate_answer = raw_answer_output.strip()
            judge_messages = build_qa_judge_messages(item, candidate_answer, media)
            raw_judge_output = ""
            judge: dict[str, Any] = {}
            last_parse_error: Exception | None = None
            max_attempts = max(1, 1 + int(args.judge_parse_error_retries))
            for attempt_idx in range(max_attempts):
                current_messages = [dict(message) for message in judge_messages]
                if attempt_idx > 0:
                    current_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Retry because the previous judgment was not valid score JSON. "
                                        "Output exactly this JSON shape and nothing else:\n"
                                        '{"score": 0.000, "reason": "One short sentence."}'
                                    ),
                                }
                            ],
                        }
                    )
                raw_judge_output = await asyncio.to_thread(
                    common.call_azure_responses_sync,
                    messages=current_messages,
                    model_cfg=judge_model_cfg,
                    timeout=args.timeout,
                    max_output_tokens=args.judge_max_output_tokens,
                )
                try:
                    judge = parse_judge_output(raw_judge_output)
                    judge["attempt_count"] = attempt_idx + 1
                    break
                except Exception as exc:
                    last_parse_error = exc
            else:
                assert last_parse_error is not None
                judge = {
                    "score": -1.0,
                    "score_band": "parse_error",
                    "reason": f"{type(last_parse_error).__name__}: {last_parse_error}",
                    "score_schema": SCORE_SCHEMA,
                    "parse_error": True,
                    "attempt_count": max_attempts,
                }
        return {
            "split": "qa",
            "task_name": item.task_name,
            "sample_id": item.sample_id,
            "sample_index": item.sample_index,
            "source_file": str(item.source_file),
            "question": item.question,
            "reference_answer": item.reference_answer,
            "candidate_answer": candidate_answer,
            "answer_model": answer_model_cfg.model,
            "judge_model": judge_model_cfg.model,
            "judge": judge,
            "raw_answer_model_output": raw_answer_output,
            "raw_judge_model_output": raw_judge_output,
            "media": common.media_meta(media),
        }


def summarize_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["judge"]["score"])
        for row in rows
        if isinstance(row.get("judge"), dict)
        and not row["judge"].get("parse_error")
        and isinstance(row["judge"].get("score"), (int, float))
        and SCORE_MIN <= float(row["judge"]["score"]) <= SCORE_MAX
    ]
    band_counts: dict[str, int] = {
        "[0.000,0.250)": 0,
        "[0.250,0.500)": 0,
        "[0.500,0.750)": 0,
        "[0.750,1.000]": 0,
        "parse_error": 0,
    }
    for row in rows:
        judge = row.get("judge") or {}
        if judge.get("parse_error") or not isinstance(judge.get("score"), (int, float)):
            band_counts["parse_error"] += 1
        else:
            score = float(judge["score"])
            if SCORE_MIN <= score <= SCORE_MAX:
                band_counts[score_band_label(score)] += 1
            else:
                band_counts["parse_error"] += 1
    total = len(rows)
    return {
        "total": total,
        "valid_score_count": len(scores),
        "parse_error_count": band_counts["parse_error"],
        "average_score": round(sum(scores) / total, 4) if total else 0.0,
        "average_score_valid_only": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "score_band_counts": band_counts,
        "score_schema": SCORE_SCHEMA,
    }


async def run_qa_split(
    items: list[common.BenchmarkItem],
    answer_model_cfg: common.ModelConfig,
    judge_model_cfg: common.ModelConfig,
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
        task_dir = output_dir / "qa" / task_name
        result_path = task_dir / "qa_judge_results.jsonl"
        if result_path.exists() and not args.resume and not args.overwrite:
            raise SystemExit(
                f"Refusing to append duplicate results to existing file without --resume or --overwrite: {result_path}"
            )
        if args.resume:
            done_ids = common.existing_result_ids(result_path)
            task_items = [item for item in task_items if item.sample_id not in done_ids]

        print(f"[qa] {task_name}: running {len(task_items)} item(s)")
        if task_items:
            coros = [run_qa_item(item, answer_model_cfg, judge_model_cfg, args, semaphore) for item in task_items]
            rows = [await coro for coro in asyncio.as_completed(coros)]
            rows.sort(key=lambda row: int(row.get("sample_index", 0)))
            common.append_jsonl(result_path, rows)

        all_rows = common.load_result_rows(result_path)
        if not all_rows:
            continue
        duplicates = common.duplicate_sample_ids(all_rows)
        if duplicates:
            raise SystemExit(f"Duplicate sample_id rows in {result_path}: {duplicates[:10]}")
        task_summary = summarize_qa(all_rows)
        task_summary.update(
            {
                "split": "qa",
                "task_name": task_name,
                "result_path": str(result_path),
                "answer_model": answer_model_cfg.model,
                "judge_model": judge_model_cfg.model,
            }
        )
        common.write_json(task_dir / "summary.json", task_summary)
        split_summary[task_name] = task_summary
    return split_summary


def build_overall_summary(qa_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    total = sum(v.get("total", 0) for v in qa_summary.values())
    valid = sum(v.get("valid_score_count", 0) for v in qa_summary.values())
    parse_errors = sum(v.get("parse_error_count", 0) for v in qa_summary.values())
    weighted_score_sum = sum(v.get("average_score", 0.0) * v.get("total", 0) for v in qa_summary.values())
    weighted_valid_score_sum = sum(
        v.get("average_score_valid_only", 0.0) * v.get("valid_score_count", 0)
        for v in qa_summary.values()
    )
    return {
        "output_dir": str(output_dir),
        "split": "qa",
        "total": total,
        "valid_score_count": valid,
        "parse_error_count": parse_errors,
        "average_score": round(weighted_score_sum / total, 4) if total else 0.0,
        "average_score_valid_only": round(weighted_valid_score_sum / valid, 4) if valid else 0.0,
        "score_schema": SCORE_SCHEMA,
        "by_task": qa_summary,
    }


async def async_main() -> None:
    args = parse_args()
    args.benchmark_data_root = args.benchmark_data_root.expanduser().resolve()
    args.media_cache_dir = args.media_cache_dir.expanduser().resolve()
    args.media_cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.benchmark_data_root.exists():
        raise SystemExit(f"Benchmark data root not found: {args.benchmark_data_root}")

    registry = common.read_registry(args.registry.expanduser().resolve())
    answer_model_cfg = common.resolve_model_config(
        args.model,
        registry,
        reasoning_effort_override=args.reasoning_effort,
        max_output_tokens_override=args.max_output_tokens,
    )
    judge_model_cfg = common.resolve_model_config(
        args.judge_model,
        registry,
        reasoning_effort_override=args.judge_reasoning_effort,
        max_output_tokens_override=args.judge_max_output_tokens,
    )
    prompt_module_check = (
        {"module_check_passed": False, "skipped": True}
        if args.skip_prompt_module_check
        else common.validate_prompt_module()
    )

    if args.output_dir is None:
        answer_alias = common.sanitize_name(answer_model_cfg.alias)
        judge_alias = common.sanitize_name(judge_model_cfg.alias)
        output_dir = common.DEFAULT_OUTPUT_ROOT / f"qa_rubric__{answer_alias}__judge_{judge_alias}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        common.init_azure_responses_client()

    manifest = {
        "benchmark_data_root": str(args.benchmark_data_root),
        "split": "qa",
        "tasks": args.tasks,
        "answer_model": answer_model_cfg.__dict__,
        "judge_model": judge_model_cfg.__dict__,
        "image_max_count": args.image_max_count,
        "video_max_frames": args.video_max_frames,
        "video_clip_frames": args.video_clip_frames,
        "dry_run": args.dry_run,
        "prompt_module_check": prompt_module_check,
    }
    common.write_json(output_dir / "manifest.json", manifest)

    all_items = common.load_items(args.benchmark_data_root, "qa", args.tasks)
    items = common.task_selected_items(
        all_items,
        limit_per_task=args.limit_per_task,
        max_items_total=args.max_items_total,
    )
    print(f"[qa] selected {len(items)} item(s) from {len(all_items)}")
    qa_summary = await run_qa_split(items, answer_model_cfg, judge_model_cfg, output_dir, args)
    overall = build_overall_summary(qa_summary, output_dir)
    common.write_json(output_dir / "summary.json", overall)
    print(f"[done] QA rubric summary: {output_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
