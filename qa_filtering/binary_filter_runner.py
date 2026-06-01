from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

try:
    from .evidence_frames import resolve_evidence_frames
    from .judge_api import call_chat_completion, create_chat_client
    from .qa_filter_io import attach_filter_metadata, load_samples, preflight_sample, write_jsonl
except ImportError:
    from evidence_frames import resolve_evidence_frames
    from judge_api import call_chat_completion, create_chat_client
    from qa_filter_io import attach_filter_metadata, load_samples, preflight_sample, write_jsonl


BuildMessages = Callable[[Any, list[str]], list[dict[str, Any]]]
ParseResponse = Callable[..., dict[str, Any]]


def _preflight_decision(kind: str, sample: Any, issues: list[str]) -> dict[str, Any]:
    if kind == "physical":
        return {
            "decision": "REJECT",
            "precondition_validity": "fail",
            "causal_dependency": "fail",
            "state_transition": "fail",
            "timeline_consistency": "fail",
            "hallucinated_effects": [],
            "physical_feasibility": "fail",
            "qa_alignment": "fail",
            "failure_tags": ["preflight_error", *issues],
            "rationale": "Preflight rejected the row before model judging: " + ", ".join(issues),
            "parse_error": False,
            "task_name": sample.task_name,
        }
    return {
        "decision": "REJECT",
        "visual_grounding": "fail",
        "logical_coherence": "fail",
        "qa_alignment": "fail",
        "failure_tags": ["preflight_error", *issues],
        "rationale": "Preflight rejected the row before model judging: " + ", ".join(issues),
        "parse_error": False,
        "task_name": sample.task_name,
    }


def _runtime_decision(kind: str, sample: Any, exc: Exception) -> dict[str, Any]:
    if kind == "physical":
        return {
            "decision": "REJECT",
            "precondition_validity": "fail",
            "causal_dependency": "fail",
            "state_transition": "fail",
            "timeline_consistency": "fail",
            "hallucinated_effects": [],
            "physical_feasibility": "fail",
            "qa_alignment": "fail",
            "failure_tags": ["judge_runtime_error"],
            "rationale": f"Filter runtime error: {type(exc).__name__}: {exc}",
            "parse_error": True,
            "task_name": sample.task_name,
        }
    return {
        "decision": "REJECT",
        "visual_grounding": "fail",
        "logical_coherence": "fail",
        "qa_alignment": "fail",
        "failure_tags": ["judge_runtime_error"],
        "rationale": f"Filter runtime error: {type(exc).__name__}: {exc}",
        "parse_error": True,
        "task_name": sample.task_name,
    }


def judge_one(
    sample: Any,
    args: Any,
    *,
    filter_name: str,
    metadata_key: str,
    kind: str,
    build_messages: BuildMessages,
    parse_response: ParseResponse,
) -> dict[str, Any]:
    raw_response = ""
    usage: dict[str, int] = {}
    frame_count = 0
    issues = preflight_sample(sample, require_source_context=not args.allow_partial_source_context)
    if issues:
        decision = _preflight_decision(kind, sample, issues)
    else:
        try:
            frames = resolve_evidence_frames(sample)
            frame_count = len(frames)
            messages = build_messages(sample, frames)
            client = create_chat_client()
            raw_response, usage = call_chat_completion(
                client,
                messages=messages,
                model=args.judge_model,
                max_tokens=args.max_tokens,
                temperature=0.0,
                reasoning_effort=getattr(args, "reasoning_effort", None),
                timeout=args.timeout,
            )
            decision = parse_response(raw_response, task_name=sample.task_name)
        except Exception as exc:
            decision = _runtime_decision(kind, sample, exc)

    meta = {
        "filter_name": filter_name,
        "judge_model": args.judge_model,
        "decision": decision.get("decision", "REJECT"),
        "failure_tags": decision.get("failure_tags", []),
        "rationale": decision.get("rationale", ""),
        "parse_error": bool(decision.get("parse_error")),
        "evidence_frame_count": frame_count,
        "usage": usage,
    }
    for key, value in decision.items():
        if key not in meta and key not in {"task_name", "parse_error"}:
            meta[key] = value
    return {
        "task_name": sample.task_name,
        "sample_id": sample.sample_id,
        "row_index": sample.row_index,
        "source_file": str(sample.source_file),
        "decision": meta["decision"],
        "judge": decision,
        "raw_judge_response": raw_response,
        "evidence_frame_count": frame_count,
        "usage": usage,
        "output_row": attach_filter_metadata(sample.row, metadata_key, meta),
    }


def run_binary_filter(
    args: Any,
    *,
    filter_name: str,
    metadata_key: str,
    kind: str,
    build_messages: BuildMessages,
    parse_response: ParseResponse,
    decisions_filename: str,
    summary_filename: str,
) -> int:
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(Path(args.input_dir))
    if not samples:
        raise SystemExit(f"No Task_*/data.jsonl rows found under {args.input_dir}")

    accepted_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        futures = [
            pool.submit(
                judge_one,
                sample,
                args,
                filter_name=filter_name,
                metadata_key=metadata_key,
                kind=kind,
                build_messages=build_messages,
                parse_response=parse_response,
            )
            for sample in samples
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            decisions.append({k: v for k, v in result.items() if k != "output_row"})
            bucket = accepted_by_task if result["decision"] == "ACCEPT" else rejected_by_task
            bucket[result["task_name"]].append(result["output_row"])
            if idx % 25 == 0 or idx == len(futures):
                print(f"[{filter_name}] processed {idx}/{len(futures)}", flush=True)

    for task_name, rows in accepted_by_task.items():
        write_jsonl(args.output_dir / "accepted" / task_name / "data.jsonl", rows)
    for task_name, rows in rejected_by_task.items():
        write_jsonl(args.output_dir / "rejected" / task_name / "data.jsonl", rows)
    write_jsonl(args.output_dir / decisions_filename, decisions)

    counts = Counter(item["decision"] for item in decisions)
    task_summary: dict[str, dict[str, int]] = {}
    for item in decisions:
        task_summary.setdefault(item["task_name"], {"ACCEPT": 0, "REJECT": 0})
        task_summary[item["task_name"]][item["decision"]] += 1
    summary = {
        "filter_name": filter_name,
        "judge_model": args.judge_model,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "total": len(decisions),
        "accepted": int(counts.get("ACCEPT", 0)),
        "rejected": int(counts.get("REJECT", 0)),
        "task_summary": task_summary,
    }
    (args.output_dir / summary_filename).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
