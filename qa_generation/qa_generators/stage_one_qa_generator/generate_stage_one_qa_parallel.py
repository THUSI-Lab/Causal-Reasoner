




from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import random
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import generate_stage_one_qa as qa              


def _default_workers() -> int:
    try:
        return min(8, int(os.cpu_count() or 8))
    except Exception:
        return 8


@dataclass(frozen=True)
class WorkerConfig:
    input_root: str
    enabled_tasks: Tuple[str, ...]
    uniform_k: int
    head: int
    tail: int
    require_videos: bool
    attach_evidence: bool
    ffmpeg_bin: str
    build_video_prefix_clips: bool
    overwrite_video_prefix_clips: bool
    strict_prefix_video: bool
    strict_schema: bool
    min_steps: int
    use_llm: bool
    llm_tasks: Tuple[str, ...]
    llm_max_tokens: int
    llm_temperature: float
    llm_single_pass: bool
    llm_vocab_guard: str
    llm_verify: str
    llm_fallback: str
    llm_require_success: bool


_WORKER_CFG: Optional[WorkerConfig] = None
_WORKER_ENABLED_TASKS: Optional[set[str]] = None
_WORKER_LLM_TASKS: Optional[set[str]] = None
_WORKER_LLM: Optional[qa.TwoStageLlm] = None
_THREAD_LOCAL = threading.local()


def _init_worker(cfg: WorkerConfig) -> None:
    global _WORKER_CFG, _WORKER_ENABLED_TASKS, _WORKER_LLM_TASKS, _WORKER_LLM
    _WORKER_CFG = cfg
    _WORKER_ENABLED_TASKS = set(cfg.enabled_tasks or [])
    _WORKER_LLM_TASKS = set(cfg.llm_tasks or [])
    _WORKER_LLM = None
    if cfg.use_llm and _WORKER_LLM_TASKS:
        llm_cfg = qa.ApiConfig()
        if int(cfg.llm_max_tokens) > 0:
            llm_cfg.max_tokens = int(cfg.llm_max_tokens)
        llm_cfg.temperature = float(cfg.llm_temperature)
        llm = qa.TwoStageLlm(llm_cfg)
        _WORKER_LLM = llm if llm.enabled() else None


def _get_llm_for_worker() -> Optional[qa.TwoStageLlm]:
    cfg = _WORKER_CFG
    if cfg is None:
        return None
    if not bool(cfg.use_llm):
        return None
    if not (_WORKER_LLM_TASKS or set()):
        return None
    if _WORKER_LLM is not None:
        return _WORKER_LLM


    cached = getattr(_THREAD_LOCAL, "llm", None)
    if cached is not None:
        return cached

    llm_cfg = qa.ApiConfig()
    if int(cfg.llm_max_tokens) > 0:
        llm_cfg.max_tokens = int(cfg.llm_max_tokens)
    llm_cfg.temperature = float(cfg.llm_temperature)
    llm = qa.TwoStageLlm(llm_cfg)
    _THREAD_LOCAL.llm = llm if llm.enabled() else None
    return _THREAD_LOCAL.llm


def _process_one_item(item_dir: str) -> Dict[str, Any]:
    cfg = _WORKER_CFG
    enabled_tasks = _WORKER_ENABLED_TASKS
    llm_tasks = _WORKER_LLM_TASKS or set()
    if cfg is None or enabled_tasks is None:
        raise RuntimeError("Worker not initialized (missing config).")

    rel_item = qa._safe_relpath(item_dir, cfg.input_root)
    item_rng = random.Random(qa._stable_int_seed(rel_item))

    try:
        samples = qa.generate_samples_for_item(
            item_dir=item_dir,
            input_root=cfg.input_root,
            enabled_tasks=enabled_tasks,
            uniform_k=int(cfg.uniform_k),
            head=int(cfg.head),
            tail=int(cfg.tail),
            require_videos=bool(cfg.require_videos),
            attach_evidence=bool(cfg.attach_evidence),
            ffmpeg_bin=str(cfg.ffmpeg_bin),
            build_video_prefix_clips=bool(cfg.build_video_prefix_clips),
            overwrite_video_prefix_clips=bool(cfg.overwrite_video_prefix_clips),
            strict_prefix_video=bool(cfg.strict_prefix_video),
            strict_schema=bool(cfg.strict_schema),
            min_steps=int(cfg.min_steps),
            rng=item_rng,
        )
    except Exception as e:
        return {
            "status": "error",
            "rel_item": rel_item,
            "error": f"{type(e).__name__}: {e}",
        }

    llm_warning = ""
    if bool(cfg.use_llm) and llm_tasks:
        llm = _get_llm_for_worker()
        if llm is not None and llm.enabled():
            try:
                samples = qa._apply_llm(
                    samples,
                    llm,
                    set(llm_tasks),
                    two_pass=not bool(cfg.llm_single_pass),
                    fallback=str(cfg.llm_fallback),
                    require_success=bool(cfg.llm_require_success),
                    vocab_guard=str(cfg.llm_vocab_guard),
                    verify=str(cfg.llm_verify),
                )
            except Exception as e:                    
                llm_warning = f"{type(e).__name__}: {e}"

    return {
        "status": "done",
        "rel_item": rel_item,
        "samples": samples,
        "llm_warning": llm_warning,
    }


def _write_audit_reports(*, output_dir: str, name: str, report: qa.AuditReport, issues_prefix: str) -> None:
    try:
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    except Exception as e:                    
        qa._log_issue(
            output_dir=output_dir,
            prefix=issues_prefix,
            severity="warn",
            phase=f"write_{name}",
            message=f"Failed to write {name}: {type(e).__name__}: {e}",
            exc=e,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel generator for Mani-LongVideo QA dataset (canonical single-step-compatible subset) "
            "from causal_plan_with_keyframes.json (final schema)."
        )
    )
    parser.add_argument("--input-root", required=True, help="Dataset root containing many item dirs with causal_plan_with_keyframes.json.")
    parser.add_argument("--output-dir", required=True, help="Output root directory (will create one folder per task with data.jsonl).")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--keep-going",
        dest="keep_going",
        action="store_true",
        help="Never abort generation due to item-level errors/audit/require checks; always save outputs and write issues JSON (default).",
    )
    run_mode.add_argument(
        "--fail-fast",
        dest="keep_going",
        action="store_false",
        help="Abort on errors / return non-zero exit codes (strict-failure behavior).",
    )
    parser.set_defaults(keep_going=True)
    parser.add_argument(
        "--issues-prefix",
        default="issues",
        help="Prefix for issue logs under output_dir (writes <prefix>_errors.jsonl/json and <prefix>_warnings.jsonl/json).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Parallelize across item dirs with this many workers (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--executor",
        choices=["auto", "process", "thread"],
        default="auto",
        help="Concurrency backend (default: auto = try process, fallback to thread).",
    )
    out_mode = parser.add_mutually_exclusive_group()
    out_mode.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output_dir task files if they already exist (only use this when intentionally merging runs).",
    )
    out_mode.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete output_dir before writing new outputs (destructive).",
    )
    out_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing output_dir (skip items already recorded in resume_state.jsonl).",
    )
    parser.add_argument(
        "--resume-allow-mismatch",
        action="store_true",
        help="Allow --resume even if run_config.json differs (may mix settings in one output_dir).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Write text-only QA (omit image/video paths and meta.evidence_files); also skips evidence existence checks in --audit.",
    )
    parser.add_argument(
        "--meta-abs-paths",
        action="store_true",
        help="Also store absolute paths in meta (meta.item_dir_abs/meta.source_path_abs/meta.input_root_abs) for easier later alignment; may reduce portability.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N item dirs (0 = no limit).")
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Fail if an item has fewer than N steps (two-stage data is typically single-step, so N=1 is common).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(qa.DEFAULT_TASKS),
        help="Subset of task names to generate (default: canonical two-stage single-step task subset).",
    )
    parser.add_argument("--uniform-k", type=int, default=8, help="Number of uniform frames for images_uniform_scene tasks (default: 8).")
    parser.add_argument("--head", type=int, default=4, help="Head frames for Task_19 (default: 4).")
    parser.add_argument("--tail", type=int, default=4, help="Tail frames for Task_19 (default: 4).")
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="If set, require video_clip assets for Task_03 (skip keyframe fallback). Ignored in --text-only mode.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable path (used by --build-video-prefix-clips).",
    )
    parser.add_argument(
        "--build-video-prefix-clips",
        action="store_true",
        help=(
            "If set, build missing video_prefix assets under <item_dir>/cumulative_last_frame_segments/ "
            "by concatenating step clips via ffmpeg (kept for CLI compatibility; typically unused for two-stage single-step items)."
        ),
    )
    parser.add_argument(
        "--overwrite-video-prefix-clips",
        action="store_true",
        help="If set, overwrite existing video_prefix assets when --build-video-prefix-clips is enabled.",
    )
    parser.add_argument(
        "--strict-prefix-video",
        action="store_true",
        help="Require true cumulative prefix clips for video_prefix tasks; do not fall back to the last-step clip.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--strict-schema",
        dest="strict_schema",
        action="store_true",
        help="Enable strict final-schema validation for causal_plan_with_keyframes.json (default).",
    )
    schema_group.add_argument(
        "--no-strict-schema",
        dest="strict_schema",
        action="store_false",
        help="Disable strict schema validation (not recommended for final generation).",
    )
    parser.set_defaults(strict_schema=True)
    parser.add_argument("--no-api", action="store_true", help="Disable OpenAI-compatible API two-stage rewriting; keep deterministic answers.")
    parser.add_argument(
        "--llm-tasks",
        nargs="*",
        default=list(qa.DEFAULT_LLM_TASKS),
        help="Tasks to rewrite/polish via API (default: all tasks).",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=0, help="Override MAX_TOKENS for API calls (0 uses env/default).")
    parser.add_argument("--llm-temperature", type=float, default=0.3, help="Override TEMPERATURE for API calls (default: 0.3).")
    parser.add_argument("--llm-single-pass", action="store_true", help="Use a single API pass (no second polishing pass).")
    parser.add_argument(
        "--llm-require-success",
        action="store_true",
        help=(
            "If set, treat any LLM call failure as an item-level error (prevents silently falling back to deterministic drafts). "
            "Recommended for high-stakes large-scale generation."
        ),
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail fast if LLM polishing is requested but the API client is not available (missing API_KEY, etc.).",
    )
    parser.add_argument(
        "--llm-fallback",
        choices=["draft", "skip", "fail"],
        default="draft",
        help="If the LLM output is rejected by safety checks, use draft (default), skip the sample, or fail the run.",
    )
    parser.add_argument(
        "--llm-vocab-guard",
        choices=["strict", "warn", "off"],
        default="off",
        help=(
            "Novel-terms vocabulary guard for LLM outputs. "
            "strict=reject when non-trivial terms are not present in the source JSON/draft; "
            "warn=allow but report in logs; off=disable this check."
        ),
    )
    parser.add_argument(
        "--llm-verify",
        choices=["strict", "warn", "off"],
        default="warn",
        help=(
            "LLM output verification mode. "
            "strict=reject on any grounding/format drift; "
            "warn=log drift but keep the output (default; recommended for large-scale runs); "
            "off=disable most checks (still rejects empty, prompt echo, frame leakage, and malformed repair answers)."
        ),
    )
    parser.add_argument(
        "--audit",
        dest="audit",
        action="store_true",
        default=True,
        help=(
            "Run built-in QA audit on output_dir after generation (default: on). "
            "Errors are logged; use --audit-strict to make audit errors fail the run."
        ),
    )
    parser.add_argument(
        "--no-audit",
        dest="audit",
        action="store_false",
        help="Disable built-in QA audit on output_dir after generation.",
    )
    parser.add_argument(
        "--audit-deep",
        dest="audit_deep",
        action="store_true",
        default=True,
        help=(
            "Run deep grounding audit (default: on; verifies Q/A against meta.source_path JSON fields; no evidence checks). "
            "Errors are logged; use --audit-strict to make deep-audit errors fail the run."
        ),
    )
    parser.add_argument(
        "--no-audit-deep",
        dest="audit_deep",
        action="store_false",
        help="Disable deep grounding audit on output_dir after generation.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run built-in QA audit on output_dir; do not generate new samples.",
    )
    parser.add_argument(
        "--audit-max-issues",
        type=int,
        default=50,
        help="Maximum number of audit issues to print (default: 50).",
    )
    parser.add_argument(
        "--audit-strict",
        action="store_true",
        help="If set, treat audit/deep-audit errors as fatal (exit code 2). Default: log errors but continue.",
    )
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help="Fail if any enabled task produces 0 samples overall.",
    )
    parser.add_argument(
        "--require-all-tasks-per-item",
        action="store_true",
        help=(
            "Fail if any item does not produce at least one sample for every enabled task. "
            "Incomplete items are reported in output_dir/incomplete_items.json (samples are still written)."
        ),
    )

    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir)
    issues_prefix = str(args.issues_prefix or "issues").strip() or "issues"
    keep_going = bool(args.keep_going)
    issue_paths = qa._issues_paths(output_dir=output_dir, prefix=issues_prefix)
    try:
        os.makedirs(output_dir, exist_ok=True)
        open(issue_paths["errors_jsonl"], "a", encoding="utf-8").close()
        open(issue_paths["warnings_jsonl"], "a", encoding="utf-8").close()
    except Exception:
        pass

    def _atexit_finalize_issues() -> None:
        try:
            qa._finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
        except Exception:
            pass

    atexit.register(_atexit_finalize_issues)

    attach_evidence = not bool(args.text_only)
    enabled_tasks = set(args.tasks or [])
    llm_tasks = set(args.llm_tasks or [])


    if bool(args.audit_only):
        audit_only_exit_code = 0

        def _audit_issue_sink(iss: qa.AuditIssue) -> None:
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        def _deep_audit_issue_sink(iss: qa.AuditIssue) -> None:
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="deep_audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        if bool(args.audit):
            rep = qa._audit_output_dir(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                require_evidence=bool(attach_evidence),
                issue_sink=_audit_issue_sink,
            )
            _write_audit_reports(output_dir=output_dir, name="audit_report.json", report=rep, issues_prefix=issues_prefix)
            if rep.errors and bool(args.audit_strict) and not keep_going:
                audit_only_exit_code = max(audit_only_exit_code, 2)
        if bool(args.audit_deep):
            rep2 = qa._audit_output_dir_deep(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                issue_sink=_deep_audit_issue_sink,
            )
            _write_audit_reports(output_dir=output_dir, name="deep_audit_report.json", report=rep2, issues_prefix=issues_prefix)
            if rep2.errors and bool(args.audit_strict) and not keep_going:
                audit_only_exit_code = max(audit_only_exit_code, 2)

        try:
            qa._finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
        except Exception:
            pass
        if audit_only_exit_code and not keep_going:
            raise SystemExit(audit_only_exit_code)
        return

    unknown = sorted([t for t in enabled_tasks if t not in set(qa.ALL_TASKS)])
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    unknown_llm = sorted([t for t in llm_tasks if t not in set(qa.ALL_TASKS)])
    if unknown_llm:
        raise ValueError(f"Unknown llm task names: {unknown_llm}")

    existing = sorted(glob.glob(os.path.join(output_dir, "*", "data.jsonl")))
    if existing:
        if bool(args.overwrite_output_dir):
            shutil.rmtree(output_dir)
        elif not (bool(args.append) or bool(args.resume)):
            raise SystemExit(
                f"Refusing to write to a non-empty output_dir (found {len(existing)} existing task data.jsonl). "
                "Use a fresh --output-dir, or pass --resume to continue from a previous run, or pass --append to merge runs, "
                "or --overwrite-output-dir to delete it."
            )

    if bool(args.build_video_prefix_clips) and not qa._ffmpeg_exists(str(args.ffmpeg_bin)):
        qa.logger.warning(
            f"ffmpeg not found for --ffmpeg-bin={args.ffmpeg_bin!r}; clip building flags will not work. "
            "Install ffmpeg or set --ffmpeg-bin, or rely on the script's video fallbacks."
        )
        if bool(args.strict_prefix_video):
            qa.logger.warning("Strict prefix-video flag is enabled; some tasks may be skipped without prebuilt prefix clips.")


    use_llm = (not bool(args.no_api)) and bool(llm_tasks)
    if use_llm:
        llm_probe = qa.TwoStageLlm(qa.ApiConfig())
        if not llm_probe.enabled():
            use_llm = False
            qa.logger.info("LLM disabled (missing API_KEY or client init failure); proceeding without API rewriting.")

    if bool(args.require_llm):
        if bool(args.no_api):
            msg = "--require-llm is set but --no-api disables LLM polishing."
            qa._log_issue(output_dir=output_dir, prefix=issues_prefix, severity="error", phase="require_llm", message=msg)
            if not bool(keep_going):
                raise SystemExit(msg)
            qa.logger.warning(msg + " keep-going is enabled; proceeding without LLM polishing.")
            use_llm = False
        elif not use_llm:
            msg = "--require-llm is set but the LLM client is not enabled (check API_KEY/API_BASE_URL/MODEL_NAME)."
            qa._log_issue(output_dir=output_dir, prefix=issues_prefix, severity="error", phase="require_llm", message=msg)
            if not bool(keep_going):
                raise SystemExit(msg)
            qa.logger.warning(msg + " keep-going is enabled; proceeding without LLM polishing.")
            use_llm = False


    run_config = qa._build_taskslist_run_config(
        input_root=input_root,
        enabled_tasks=sorted(enabled_tasks),
        text_only=bool(args.text_only),
        meta_abs_paths=bool(args.meta_abs_paths),
        uniform_k=int(args.uniform_k),
        head=int(args.head),
        tail=int(args.tail),
        require_videos=bool(args.require_videos),
        ffmpeg_bin=str(args.ffmpeg_bin),
        build_video_prefix_clips=bool(args.build_video_prefix_clips),
        overwrite_video_prefix_clips=bool(args.overwrite_video_prefix_clips),
        strict_prefix_video=bool(args.strict_prefix_video),
        strict_schema=bool(args.strict_schema),
        min_steps=int(args.min_steps),
        llm_enabled=bool(use_llm),
        llm_tasks=sorted(llm_tasks),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_temperature=float(args.llm_temperature),
        llm_single_pass=bool(args.llm_single_pass),
        llm_vocab_guard=str(args.llm_vocab_guard),
        llm_verify=str(args.llm_verify),
        llm_fallback=str(args.llm_fallback),
        llm_require_success=bool(args.llm_require_success),
    )
    cfg_path = qa._run_config_path(output_dir)
    prev_cfg = qa._load_json_maybe(cfg_path)
    if prev_cfg is None:
        if not existing or bool(args.resume):
            if bool(args.resume) and existing and not bool(args.overwrite_output_dir):
                qa.logger.warning(f"--resume is set but missing {cfg_path}; creating it from current args (cannot verify existing outputs).")
            qa._write_json_atomic(cfg_path, run_config)
        elif bool(args.append):
            qa.logger.warning(f"--append is set but missing {cfg_path}; not writing run_config.json for a merged output_dir.")
    else:
        if bool(args.resume) and not bool(args.resume_allow_mismatch):
            prev_cfg_dict = prev_cfg if isinstance(prev_cfg, dict) else {}
            mism = qa._run_config_mismatch_keys(prev_cfg_dict, run_config)
            if mism:
                details = "; ".join([f"{k}: old={prev_cfg_dict.get(k)!r} new={run_config.get(k)!r}" for k in mism[:12]])
                more = "" if len(mism) <= 12 else f" (+{len(mism) - 12} more)"
                raise SystemExit(
                    "--resume refused due to run_config.json mismatch. "
                    + details
                    + more
                    + ". Use a fresh --output-dir, or use --append to intentionally merge runs, or pass --resume-allow-mismatch."
                )

    item_dirs = qa._list_item_dirs(input_root)
    if args.limit and int(args.limit) > 0:
        item_dirs = item_dirs[: int(args.limit)]
    if not item_dirs:
        raise FileNotFoundError(f"No item dirs found under {input_root} (expecting causal_plan_with_keyframes.json).")

    qa.logger.info(f"Found {len(item_dirs)} item dirs under: {input_root}")
    qa.logger.info(f"Enabled tasks: {sorted(enabled_tasks)}")
    if llm_tasks:
        qa.logger.info(f"LLM rewrite tasks: {sorted(llm_tasks)}")

    resume_state = qa._resume_state_path(output_dir)
    processed_items: set[str] = set()
    if bool(args.resume):
        resume_recs = qa._load_resume_state(resume_state)
        processed_items = {k for k, rec in resume_recs.items() if isinstance(rec, dict) and rec.get("status") == "done"}

        if not processed_items and not resume_recs and existing and not bool(args.overwrite_output_dir):
            inferred = qa._infer_processed_items_from_output_dir(output_dir, sorted(enabled_tasks))
            if inferred:
                processed_items = set(inferred)
                qa.logger.info(f"Resume: inferred processed items from existing outputs: {len(processed_items)}")
        if processed_items:
            before = len(item_dirs)
            item_dirs = [d for d in item_dirs if qa._safe_relpath(d, input_root) not in processed_items]
            qa.logger.info(f"Resume: filtered item dirs: {before} -> {len(item_dirs)}")


    for t in sorted(enabled_tasks):
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        open(os.path.join(output_dir, t, "data.jsonl"), "a", encoding="utf-8").close()

    worker_cfg = WorkerConfig(
        input_root=input_root,
        enabled_tasks=tuple(sorted(enabled_tasks)),
        uniform_k=int(args.uniform_k),
        head=int(args.head),
        tail=int(args.tail),
        require_videos=bool(args.require_videos),
        attach_evidence=bool(attach_evidence),
        ffmpeg_bin=str(args.ffmpeg_bin),
        build_video_prefix_clips=bool(args.build_video_prefix_clips),
        overwrite_video_prefix_clips=bool(args.overwrite_video_prefix_clips),
        strict_prefix_video=bool(args.strict_prefix_video),
        strict_schema=bool(args.strict_schema),
        min_steps=int(args.min_steps),
        use_llm=bool(use_llm),
        llm_tasks=tuple(sorted(llm_tasks)),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_temperature=float(args.llm_temperature),
        llm_single_pass=bool(args.llm_single_pass),
        llm_vocab_guard=str(args.llm_vocab_guard),
        llm_verify=str(args.llm_verify),
        llm_fallback=str(args.llm_fallback),
        llm_require_success=bool(args.llm_require_success),
    )

    total = 0
    per_task_counts: Dict[str, int] = {}
    prefix_fallback_samples = 0
    incomplete_items: List[Dict[str, Any]] = []
    incomplete_count = 0
    skipped_items = 0
    exit_code = 0

    def _handle_result(res: Dict[str, Any]) -> None:
        nonlocal total, prefix_fallback_samples, incomplete_count, skipped_items, exit_code
        status = str(res.get("status") or "")
        rel_item = str(res.get("rel_item") or "")
        if not rel_item:
            rel_item = "<unknown>"

        if status != "done":
            skipped_items += 1
            msg = str(res.get("error") or "unknown error")
            qa.logger.error(f"ITEM_ERROR item={rel_item} status={status} error={msg}")
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="item_generation",
                message=f"Item generation failed: {msg}",
                rel_item=rel_item,
            )
            if bool(args.require_all_tasks_per_item):
                incomplete_items.append({"item_dir": rel_item, "missing_tasks": sorted(enabled_tasks), "error": msg})
            qa._append_jsonl(
                resume_state,
                {"rel_item": rel_item, "status": "error", "error": msg},
            )
            return

        llm_warning = str(res.get("llm_warning") or "").strip()
        if llm_warning:
            qa.logger.warning(f"LLM stage failed for item={rel_item}: {llm_warning}; proceeding with deterministic drafts.")
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="llm_stage",
                message=f"LLM stage failed; using deterministic drafts: {llm_warning}",
                rel_item=rel_item,
            )

        samples = res.get("samples") if isinstance(res.get("samples"), list) else []
        missing: List[str] = []
        if bool(args.require_all_tasks_per_item):
            produced = {s.task_name for s in samples if isinstance(s, qa.Sample)}
            missing = sorted([t for t in enabled_tasks if t not in produced])
            if missing:
                incomplete_items.append({"item_dir": rel_item, "missing_tasks": missing})
                incomplete_count += 1
                exit_code = max(exit_code, 2)
                qa.logger.warning(f"INCOMPLETE_ITEM item={rel_item} missing_tasks={missing}")
                qa._log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="missing_tasks_per_item",
                    message=f"Item missing enabled tasks: {missing}",
                    rel_item=rel_item,
                    details={"missing_tasks": missing},
                )

        per_item_counts: Dict[str, int] = {}
        for s in samples:
            if not isinstance(s, qa.Sample):
                continue
            per_item_counts[s.task_name] = int(per_item_counts.get(s.task_name, 0)) + 1

            if bool(attach_evidence):
                v = str(s.video or "").replace("\\", "/")
                if s.evidence_type == qa.EVIDENCE_PREFIX and v:
                    if "/cumulative_last_frame_segments/" not in v:
                        prefix_fallback_samples += 1

            entry = qa._sharegpt_entry(s, attach_evidence=bool(attach_evidence))
            if bool(args.meta_abs_paths):
                meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                meta["input_root_abs"] = os.path.abspath(input_root)
                meta["item_dir_abs"] = os.path.abspath(os.path.join(input_root, rel_item))
                meta["source_path_abs"] = (
                    qa._abs_under_root(str(s.source_path or ""), input_root) if str(s.source_path or "").strip() else ""
                )
                entry["meta"] = meta

            out_path = os.path.join(output_dir, s.task_name, "data.jsonl")
            try:
                qa._write_jsonl(out_path, entry)
            except Exception as e:                    
                qa._log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="write_jsonl",
                    message=f"Failed to write sample JSONL: {type(e).__name__}: {e}",
                    rel_item=rel_item,
                    task_name=s.task_name,
                    exc=e,
                    details={"out_path": out_path},
                )
                try:
                    qa._append_jsonl(
                        os.path.join(output_dir, f"{issues_prefix}_unsaved_samples.jsonl"),
                        {
                            "rel_item": rel_item,
                            "task_name": s.task_name,
                            "error": f"{type(e).__name__}: {e}",
                            "entry": entry,
                        },
                    )
                except Exception:
                    pass
                continue

            total += 1
            per_task_counts[s.task_name] = int(per_task_counts.get(s.task_name, 0)) + 1

        qa._append_jsonl(
            resume_state,
            {
                "rel_item": rel_item,
                "status": "done_incomplete" if missing else "done",
                "samples_written": int(len(samples)),
                "per_task_counts": per_item_counts,
                "missing_tasks": missing,
            },
        )

    exec_s = str(args.executor or "auto").strip().lower()
    if exec_s not in {"auto", "process", "thread"}:
        raise SystemExit(f"Unknown --executor: {args.executor!r}")

    def _run_items_with(kind: str, item_dirs_run: Sequence[str]) -> List[str]:
        remaining: List[str] = []
        if not item_dirs_run:
            return remaining
        if kind == "thread":
            _init_worker(worker_cfg)
            with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
                futures = {ex.submit(_process_one_item, d): d for d in item_dirs_run}
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        d = futures.pop(fut)
                        try:
                            res = fut.result()
                        except Exception as e:
                            res = {
                                "status": "error",
                                "rel_item": qa._safe_relpath(d, input_root),
                                "error": f"{type(e).__name__}: {e}",
                            }
                        _handle_result(res if isinstance(res, dict) else {"status": "error", "rel_item": qa._safe_relpath(d, input_root), "error": "worker returned non-dict"})
            return remaining


        try:
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(worker_cfg,)) as ex:
                futures = {ex.submit(_process_one_item, d): d for d in item_dirs_run}
                pending = set(item_dirs_run)
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        d = futures.pop(fut)
                        try:
                            res = fut.result()
                        except Exception as e:

                            qa.logger.warning(f"Process worker failure for item={qa._safe_relpath(d, input_root)}: {e}")
                            pending.add(d)
                            remaining = sorted(list(pending))
                            qa._log_issue(
                                output_dir=output_dir,
                                prefix=issues_prefix,
                                severity="warn",
                                phase="executor_process",
                                message=f"Process worker failure; will retry remaining items in thread mode: {type(e).__name__}: {e}",
                                rel_item=qa._safe_relpath(d, input_root),
                            )
                            return remaining
                        pending.discard(d)
                        _handle_result(
                            res if isinstance(res, dict) else {"status": "error", "rel_item": qa._safe_relpath(d, input_root), "error": "worker returned non-dict"}
                        )
            return remaining
        except Exception as e:
            qa.logger.warning(f"Process executor failed to start: {e}")
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="executor_process",
                message=f"Process executor failed; will retry in thread mode: {type(e).__name__}: {e}",
                exc=e,
            )
            return list(item_dirs_run)

    qa.logger.info(f"Parallel run start: items={len(item_dirs)} workers={int(args.workers)} executor={exec_s}")
    remaining_items: List[str] = []
    if exec_s == "thread":
        remaining_items = _run_items_with("thread", item_dirs)
    elif exec_s == "process":
        remaining_items = _run_items_with("process", item_dirs)
    else:
        remaining_items = _run_items_with("process", item_dirs)
        if remaining_items:
            remaining_items = _run_items_with("thread", remaining_items)

    if per_task_counts:
        counts_inline = ", ".join([f"{k}={per_task_counts[k]}" for k in sorted(per_task_counts.keys())])
        qa.logger.info(f"Per-task counts: {counts_inline}")
    if bool(attach_evidence) and prefix_fallback_samples:
        qa.logger.info(f"Video fallback summary: video_prefix_non_cumulative={prefix_fallback_samples}")
    qa.logger.info(f"Done. Total samples_written={total}. Output_dir={output_dir}")
    if bool(skipped_items):
        qa.logger.warning(f"Skipped items due to item-level errors: {skipped_items}/{len(item_dirs)}")
    if bool(incomplete_count):
        qa.logger.warning(f"Items with missing tasks: {incomplete_count}/{len(item_dirs)}")
    if incomplete_items:
        report_path = os.path.join(output_dir, "incomplete_items.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(incomplete_items, f, ensure_ascii=False, indent=2)
            qa.logger.warning(f"Wrote incomplete items report: {report_path}")
        except Exception as e:                    
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="write_incomplete_items",
                message=f"Failed to write incomplete_items.json: {type(e).__name__}: {e}",
                exc=e,
                details={"path": report_path},
            )

    missing_overall = sorted([t for t in enabled_tasks if not qa._jsonl_has_any_entry(os.path.join(output_dir, t, "data.jsonl"))])
    if missing_overall:
        msg = f"Some enabled tasks produced 0 samples: {missing_overall}"
        if bool(args.require_all_tasks):
            qa.logger.error(msg)
            exit_code = max(exit_code, 2)
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="missing_tasks_overall",
                message=msg,
                details={"missing_tasks": missing_overall},
            )
        else:
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="missing_tasks_overall",
                message=msg,
                details={"missing_tasks": missing_overall},
            )
        qa.logger.warning(msg)

    if incomplete_items:
        exit_code = max(exit_code, 2)
        qa._log_issue(
            output_dir=output_dir,
            prefix=issues_prefix,
            severity="error",
            phase="incomplete_items",
            message=f"Incomplete items encountered: {len(incomplete_items)}",
            details={"count": int(len(incomplete_items))},
        )

    if bool(args.audit):
        def _audit_issue_sink(iss: qa.AuditIssue) -> None:
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        try:
            report = qa._audit_output_dir(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                require_evidence=bool(attach_evidence),
                issue_sink=_audit_issue_sink,
            )
        except Exception as e:                    
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit",
                message=f"Audit failed: {type(e).__name__}: {e}",
                exc=e,
            )
            if not keep_going:
                raise
            report = qa.AuditReport(total_samples=0, errors=1, warnings=0, issues=[])
        qa.logger.info(f"Audit summary: samples={report.total_samples} errors={report.errors} warnings={report.warnings}")
        _write_audit_reports(output_dir=output_dir, name="audit_report.json", report=report, issues_prefix=issues_prefix)
        if report.errors and bool(args.audit_strict):
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_strict",
                message="Audit reported errors and --audit-strict is set.",
            )
            if not keep_going:
                exit_code = max(exit_code, 2)
            qa.logger.warning("Audit reported errors but keep-going is enabled; continuing.")
        if report.errors and not bool(args.audit_strict):
            qa.logger.warning("Audit reported errors. --audit-strict is not set; continuing.")

    if bool(args.audit_deep):
        def _deep_audit_issue_sink(iss: qa.AuditIssue) -> None:
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="deep_audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        try:
            deep = qa._audit_output_dir_deep(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                issue_sink=_deep_audit_issue_sink,
            )
        except Exception as e:                    
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="deep_audit",
                message=f"Deep audit failed: {type(e).__name__}: {e}",
                exc=e,
            )
            if not keep_going:
                raise
            deep = qa.AuditReport(total_samples=0, errors=1, warnings=0, issues=[])
        qa.logger.info(f"Deep audit summary: samples={deep.total_samples} errors={deep.errors} warnings={deep.warnings}")
        _write_audit_reports(output_dir=output_dir, name="deep_audit_report.json", report=deep, issues_prefix=issues_prefix)
        if deep.errors and bool(args.audit_strict):
            qa._log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_strict",
                message="Deep audit reported errors and --audit-strict is set.",
            )
            if not keep_going:
                exit_code = max(exit_code, 2)
            qa.logger.warning("Deep audit reported errors but keep-going is enabled; continuing.")
        if deep.errors and not bool(args.audit_strict):
            qa.logger.warning("Deep audit reported errors. --audit-strict is not set; continuing.")

    if exit_code:
        if not keep_going:
            try:
                qa._finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
            except Exception:
                pass
            raise SystemExit(exit_code)
        qa.logger.warning(f"Completed with exit_code={exit_code} but keep-going is enabled; outputs are preserved.")

    try:
        qa._finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
    except Exception:
        pass


if __name__ == "__main__":
    main()
