
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .evidence_frames import resolve_evidence_frames
    from .judge_api import call_chat_completion, create_chat_client
    from .qa_filter_io import attach_filter_metadata, load_samples, preflight_sample, write_jsonl
    from .qwen_two_axis_score_core import build_qwen_two_axis_score_messages, parse_qwen_two_axis_score_response
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evidence_frames import resolve_evidence_frames
    from judge_api import call_chat_completion, create_chat_client
    from qa_filter_io import attach_filter_metadata, load_samples, preflight_sample, write_jsonl
    from qwen_two_axis_score_core import build_qwen_two_axis_score_messages, parse_qwen_two_axis_score_response


SCORER_NAME = "qwen_two_axis_score_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score QA rows with Qwen3.5-397B-A17B visual-grounding and logical-coherence axes, then retain the top-ranked fraction.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Input QA dir containing Task_*/data.jsonl.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dir for accepted/, rejected/, and score summaries.")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--top-fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "Qwen3.5-397B-A17B"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--allow-non-qwen-judge", action="store_true")
    parser.add_argument("--allow-partial-source-context", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _hash_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preflight_score(sample: Any, issues: list[str]) -> dict[str, Any]:
    return {
        "visual_grounding_score": 0.0,
        "logical_coherence_score": 0.0,
        "overall_score": 0.0,
        "visual_grounding_rationale": "",
        "logical_coherence_rationale": "",
        "score_confidence": "low",
        "major_failure_tags": ["preflight_error", *issues],
        "score_formula": "invalid/preflight rows score 0",
        "score_warnings": issues,
        "parse_error": False,
        "task_name": sample.task_name,
    }


def score_one(sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    input_sha = _hash_json(sample.row)
    raw_response = ""
    usage: dict[str, int] = {}
    evidence_frame_count = 0
    issues = preflight_sample(sample, require_source_context=not args.allow_partial_source_context)
    if issues:
        score = _preflight_score(sample, issues)
    else:
        try:
            frames = resolve_evidence_frames(sample)
            evidence_frame_count = len(frames)
            messages = build_qwen_two_axis_score_messages(sample, frames)
            client = create_chat_client()
            raw_response, usage = call_chat_completion(
                client,
                messages=messages,
                model=args.judge_model,
                max_tokens=args.max_tokens,
                temperature=0.0,
                timeout=args.timeout,
            )
            score = parse_qwen_two_axis_score_response(raw_response, task_name=sample.task_name)
        except Exception as exc:
            raw_response = raw_response or ""
            score = {
                "visual_grounding_score": 0.0,
                "logical_coherence_score": 0.0,
                "overall_score": 0.0,
                "visual_grounding_rationale": "",
                "logical_coherence_rationale": "",
                "score_confidence": "low",
                "major_failure_tags": ["judge_runtime_error"],
                "score_formula": "runtime failures score 0",
                "score_warnings": [f"{type(exc).__name__}: {exc}"],
                "parse_error": True,
                "task_name": sample.task_name,
            }
    return {
        "task_name": sample.task_name,
        "sample_id": sample.sample_id,
        "row_index": sample.row_index,
        "source_file": str(sample.source_file),
        "input_sha256": input_sha,
        "judge_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest() if raw_response else "",
        "score": score,
        "raw_judge_response": raw_response,
        "evidence_frame_count": evidence_frame_count,
        "usage": usage,
        "row": sample.row,
    }


def _selected_count(total: int, top_fraction: float, top_k: int) -> int:
    if total <= 0:
        return 0
    if top_k > 0:
        return min(total, top_k)
    keep = int(round(total * top_fraction))
    return max(1, min(total, keep))


def main() -> int:
    args = parse_args()
    if "qwen" not in args.judge_model.lower() and not args.allow_non_qwen_judge:
        raise SystemExit("Qwen scorer requires a Qwen judge model unless --allow-non-qwen-judge is passed.")
    if not (0 < args.top_fraction <= 1.0):
        raise SystemExit("--top-fraction must be in (0, 1].")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.input_dir)
    if not samples:
        raise SystemExit(f"No Task_*/data.jsonl rows found under {args.input_dir}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(score_one, sample, args) for sample in samples]
        for idx, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if idx % 25 == 0 or idx == len(futures):
                print(f"[qwen-score] processed {idx}/{len(futures)}", flush=True)

    results.sort(key=lambda x: (-float(x["score"]["overall_score"]), x["task_name"], x["sample_id"]))
    keep_n = _selected_count(len(results), args.top_fraction, args.top_k)
    selection_rule = f"top_k={keep_n}" if args.top_k > 0 else f"top_fraction={args.top_fraction}"

    accepted_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        selected = rank <= keep_n
        score_meta = {
            "scorer_name": SCORER_NAME,
            "judge_model": args.judge_model,
            "rank": rank,
            "selected": selected,
            "selection_rule": selection_rule,
            "input_sha256": item["input_sha256"],
            "judge_response_sha256": item["judge_response_sha256"],
            "evidence_frame_count": item["evidence_frame_count"],
            "usage": item["usage"],
            **item["score"],
        }
        row = attach_filter_metadata(item["row"], "qwen_two_axis_score", score_meta)
        (accepted_by_task if selected else rejected_by_task)[item["task_name"]].append(row)
        records.append({k: v for k, v in item.items() if k != "row"} | {"rank": rank, "selected": selected, "selection_rule": selection_rule})

    for task_name, rows in accepted_by_task.items():
        write_jsonl(args.output_dir / "accepted" / task_name / "data.jsonl", rows)
    for task_name, rows in rejected_by_task.items():
        write_jsonl(args.output_dir / "rejected" / task_name / "data.jsonl", rows)
    write_jsonl(args.output_dir / "qwen_two_axis_scores.jsonl", records)

    bands = Counter()
    for item in results:
        score = float(item["score"]["overall_score"])
        label = "0-24" if score < 25 else "25-49" if score < 50 else "50-74" if score < 75 else "75-89" if score < 90 else "90-100"
        bands[label] += 1
    summary = {
        "scorer_name": SCORER_NAME,
        "judge_model": args.judge_model,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "total": len(results),
        "selected": keep_n,
        "selection_rule": selection_rule,
        "score_band_counts": dict(bands),
    }
    (args.output_dir / "qwen_two_axis_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
