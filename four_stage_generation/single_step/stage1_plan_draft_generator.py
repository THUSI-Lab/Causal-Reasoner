


from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from four_stage_common import (
    _contains_frame_ref,
    _contains_time_ref,
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_api_content,
    build_retry_prefix,
    call_chat_completion,
    collect_videos,
    default_output_root,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    guard_schema_fingerprint,
    initialize_api_client,
    logger,
    normalize_draft_plan,
    read_json,
    sample_video_to_frames,
    sampling_config_from_args,
    save_sampled_frames_jpegs,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from four_stage_prompt_templates import SYSTEM_PROMPT_ANALYST, build_stage1_user_prompt

if TYPE_CHECKING:
    from four_stage_common import ApiConfig, SamplingConfig


_STAGE1_ALLOWED_TOP_KEYS = {"high_level_goal", "steps"}
_STAGE1_ALLOWED_STEP_KEYS = {
    "step_id",
    "step_goal",
    "rationale",
    "causal_chain",
    "counterfactual_challenge_question",
    "expected_challenge_outcome",
    "failure_reflecting",
}
_STAGE1_ALLOWED_CAUSAL_CHAIN_KEYS = {
    "agent",
    "action",
    "patient",
    "causal_precondition_on_spatial",
    "causal_precondition_on_affordance",
    "causal_effect_on_spatial",
    "causal_effect_on_affordance",
}
_STAGE1_ALLOWED_FAILURE_REFLECTING_KEYS = {"reason", "recovery_strategy"}
_STAGE1_FORBIDDEN_KEYS = {"critical_frames", "frame_index", "interaction", "keyframe_image_path"}


def _find_forbidden_keys(obj: Any, path: str) -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}" if path else k
            if k in _STAGE1_FORBIDDEN_KEYS:
                errors.append(f"Forbidden key '{k}' found at: {next_path}")
            errors.extend(_find_forbidden_keys(v, next_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errors.extend(_find_forbidden_keys(v, f"{path}[{i}]"))
    return errors


def _stage1_raw_schema_errors(plan: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(plan, dict):
        return ["Stage 1 output must be a JSON object."]

    errors.extend(_find_forbidden_keys(plan, ""))

    extra_top = sorted(set(plan.keys()) - _STAGE1_ALLOWED_TOP_KEYS)
    if extra_top:
        errors.append(f"Stage 1 top-level contains extra keys (not allowed): {extra_top}")

    steps = plan.get("steps")
    if not isinstance(steps, list):
        errors.append("Stage 1 'steps' must be a list.")
        return errors

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] is not an object.")
            continue
        extra_step = sorted(set(step.keys()) - _STAGE1_ALLOWED_STEP_KEYS)
        if extra_step:
            errors.append(f"steps[{i}] contains extra keys (not allowed): {extra_step}")

        cc = step.get("causal_chain")
        if cc is None or not isinstance(cc, dict):
            errors.append(f"steps[{i}].causal_chain missing/invalid (expected an object).")
        else:
            extra_cc = sorted(set(cc.keys()) - _STAGE1_ALLOWED_CAUSAL_CHAIN_KEYS)
            if extra_cc:
                errors.append(f"steps[{i}].causal_chain contains extra keys (not allowed): {extra_cc}")
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                v = cc.get(k)
                if not isinstance(v, (list, str)):
                    errors.append(f"steps[{i}].causal_chain.{k} must be a non-empty array of strings (sentence list).")
                elif isinstance(v, str) and not v.strip():
                    errors.append(f"steps[{i}].causal_chain.{k} is empty.")
                elif isinstance(v, list) and not v:
                    errors.append(f"steps[{i}].causal_chain.{k} is empty.")

        fr = step.get("failure_reflecting")
        if isinstance(fr, dict):
            extra_fr = sorted(set(fr.keys()) - _STAGE1_ALLOWED_FAILURE_REFLECTING_KEYS)
            if extra_fr:
                errors.append(f"steps[{i}].failure_reflecting contains extra keys (not allowed): {extra_fr}")
            if not (isinstance(fr.get("reason"), str) and str(fr.get("reason")).strip()):
                errors.append(f"steps[{i}].failure_reflecting.reason must be a non-empty string.")
            if not (isinstance(fr.get("recovery_strategy"), str) and str(fr.get("recovery_strategy")).strip()):
                errors.append(f"steps[{i}].failure_reflecting.recovery_strategy must be a non-empty string.")

    return errors


def _draft_hard_errors(draft: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    def _has_disallowed_ref(text: Any) -> bool:

        if isinstance(text, list):
            return any(_has_disallowed_ref(x) for x in text)
        if isinstance(text, dict):
            return any(_has_disallowed_ref(v) for v in text.values())
        return _contains_frame_ref(text) or _contains_time_ref(text)

    def _nonempty_str(v: Any) -> bool:
        return isinstance(v, str) and v.strip() != ""

    _LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•])\s+")
    _LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)、\)]\s*")

    def _nonempty_statement_list(v: Any) -> bool:
        if not isinstance(v, list) or not v:
            return False
        for item in v:
            if not isinstance(item, str):
                return False
            s = item.strip()
            if not s:
                return False
            if "\n" in s:
                return False
            if _LEADING_LIST_MARKER_RE.match(s) or _LEADING_NUMBER_RE.match(s):
                return False
            if not s.endswith("."):
                return False
        return True

    goal = str(draft.get("high_level_goal", "")).strip()
    if not goal:
        errors.append("high_level_goal is missing/empty.")
    if _has_disallowed_ref(goal):
        errors.append("high_level_goal must not reference frame indices or timestamps (e.g., 'Frame 12', 't=3.2s').")

    steps = draft.get("steps", [])
    if not isinstance(steps, list) or not steps:
        errors.append("steps is missing/empty.")
        return errors


    if len(steps) < 1:
        errors.append(f"Too few steps: got {len(steps)} (required exactly 1).")
    if len(steps) > 1:
        errors.append(f"Too many steps: got {len(steps)} (required exactly 1). Short videos must use exactly 1 step.")

    step_goals: List[str] = []
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] is not an object.")
            continue
        sg = str(step.get("step_goal", "")).strip()
        if not sg:
            errors.append(f"steps[{i}].step_goal is empty.")
        if _has_disallowed_ref(sg):
            errors.append(f"steps[{i}].step_goal must not reference frame indices or timestamps.")
        if sg.startswith("unnamed_step_"):
            errors.append(f"steps[{i}].step_goal looks like a placeholder ('{sg}').")
        step_goals.append(sg)

        rationale = step.get("rationale", "")
        if not _nonempty_str(rationale):
            errors.append(f"steps[{i}].rationale is empty.")
        if _has_disallowed_ref(rationale):
            errors.append(f"steps[{i}].rationale must not reference frame indices or timestamps.")

        cc = step.get("causal_chain")
        if not isinstance(cc, dict):
            errors.append(f"steps[{i}].causal_chain missing/invalid (expected an object).")
            continue
        for k in ("agent", "action", "patient"):
            if not _nonempty_str(cc.get(k)):
                errors.append(f"steps[{i}].causal_chain.{k} is empty.")
            if _has_disallowed_ref(cc.get(k)):
                errors.append(f"steps[{i}].causal_chain.{k} must not reference frame indices or timestamps.")
        for k in (
            "causal_precondition_on_spatial",
            "causal_precondition_on_affordance",
            "causal_effect_on_spatial",
            "causal_effect_on_affordance",
        ):
            text = cc.get(k)
            if not _nonempty_statement_list(text):
                errors.append(f"steps[{i}].causal_chain.{k} is empty/invalid (expected a non-empty list of sentences).")
            if _has_disallowed_ref(text):
                errors.append(f"steps[{i}].causal_chain.{k} must not reference frame indices or timestamps.")

        cq = step.get("counterfactual_challenge_question", "")
        co = step.get("expected_challenge_outcome", "")
        if not _nonempty_str(cq):
            errors.append(f"steps[{i}].counterfactual_challenge_question is empty.")
        if not _nonempty_str(co):
            errors.append(f"steps[{i}].expected_challenge_outcome is empty.")
        if _nonempty_str(cq) and not any(str(cq).lstrip().lower().startswith(p) for p in ("what if", "what would", "suppose", "imagine")):
            errors.append(f"steps[{i}].counterfactual_challenge_question must start with 'What if/What would/Suppose/Imagine ...?'.")
        if _has_disallowed_ref(cq) or _has_disallowed_ref(co):
            errors.append(f"steps[{i}] challenge fields must not reference frame indices or timestamps.")

        fr = step.get("failure_reflecting")
        reason = fr.get("reason") if isinstance(fr, dict) else ""
        recovery = fr.get("recovery_strategy") if isinstance(fr, dict) else ""
        if not _nonempty_str(reason):
            errors.append(f"steps[{i}].failure_reflecting.reason is empty.")
        if not _nonempty_str(recovery):
            errors.append(f"steps[{i}].failure_reflecting.recovery_strategy is empty.")
        if _has_disallowed_ref(reason) or _has_disallowed_ref(recovery):
            errors.append(f"steps[{i}].failure_reflecting must not reference frame indices or timestamps.")

    non_empty = [g for g in step_goals if g]
    if len(set(non_empty)) != len(non_empty):
        errors.append("Duplicate step_goal detected (must be unique across steps).")



    if len(steps) > 1:
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            sg = str(step.get("step_goal", "")).strip()
            and_count = len(re.findall(r'\band\b', sg, re.IGNORECASE))
            comma_count = sg.count(',')
            if and_count >= 3 or comma_count >= 5:
                errors.append(
                    f"steps[{i}].step_goal may be a mega-step ({and_count} 'and', {comma_count} commas): "
                    f"'{sg[:80]}...'. A step should center on ONE sub-goal. Split into smaller steps."
                )



    _CATCHALL_RE = re.compile(
        r'\b(various|miscellaneous|rest of the|leftover|assorted)\b'
        r'|'
        r'\bremaining\s+(small|loose|other|misc|kitchen|items|objects|things|stuff)\b',
        re.IGNORECASE,
    )
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        sg = str(step.get("step_goal", "")).strip()
        catchall_matches = _CATCHALL_RE.findall(sg)
        if catchall_matches:
            errors.append(
                f"steps[{i}].step_goal uses vague catch-all language "
                f"({', '.join(repr(m) for m in catchall_matches)}): '{sg[:80]}...'. "
                f"Name the SPECIFIC objects instead of bundling them with 'remaining/other/various'. "
                f"Split into focused steps each targeting ONE or TWO named objects."
            )


    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
        "with", "from", "into", "onto", "off", "up", "down", "out", "is",
        "are", "it", "its", "this", "that", "by", "as",
    }
    for i in range(len(steps) - 1):
        if not isinstance(steps[i], dict) or not isinstance(steps[i + 1], dict):
            continue
        s1_words = str(steps[i].get("step_goal", "")).lower().split()
        s2_words = str(steps[i + 1].get("step_goal", "")).lower().split()
        w1 = set(s1_words) - _STOP_WORDS
        w2 = set(s2_words) - _STOP_WORDS
        if w1 and w2:
            overlap = len(w1 & w2) / min(len(w1), len(w2))
            if overlap > 0.7:
                errors.append(
                    f"steps[{i + 1}] and steps[{i + 2}] have highly overlapping step_goals "
                    f"({overlap:.0%} content word overlap). Each step should describe a DISTINCT sub-task."
                )


    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        sg = str(step.get("step_goal", "")).strip()
        content_words = [w for w in sg.lower().split() if w not in _STOP_WORDS and len(w) > 2]
        if len(content_words) < 3:
            errors.append(
                f"steps[{i}].step_goal is too vague (only {len(content_words)} content words): "
                f"'{sg[:80]}'. Specify the concrete objects and actions involved."
            )

    return errors


def _can_resume_stage1(draft_path: str, manifest_path: str) -> bool:

    if not (os.path.exists(draft_path) and os.path.exists(manifest_path)):
        return False
    try:
        draft = read_json(draft_path)
        manifest = read_json(manifest_path)
    except Exception:
        return False

    steps = draft.get("steps", [])
    if not (isinstance(steps, list) and isinstance(draft.get("high_level_goal"), str)):
        return False

    for idx, st in enumerate(steps, start=1):
        if not isinstance(st, dict):
            return False
        if "critical_frames" in st or "frame_index" in st or "interaction" in st or "keyframe_image_path" in st:
            return False
        try:
            if int(st.get("step_id")) != idx:
                return False
        except Exception:
            return False

    if not (int(manifest.get("num_frames", 0)) > 0 and isinstance(manifest.get("frames"), list)):
        return False

    if _stage1_raw_schema_errors(draft):
        return False
    return not _draft_hard_errors(draft)


def run_stage1_for_video(
    video_path: str,
    output_root: str,
    api_cfg: ApiConfig,
    sampling_cfg: SamplingConfig,
    overwrite: bool,
    max_retries: int,
    *,
    allow_unfingerprinted_resume: bool = False,
) -> str:
    t_start = time.perf_counter()
    if int(getattr(sampling_cfg, "max_frames", 0) or 0) != 100:
        raise RuntimeError(
            "Stage 1 requires --max-frames=100 to keep step localization and keyframe index semantics consistent. "
            f"Got max_frames={getattr(sampling_cfg, 'max_frames', None)}."
        )
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    sampled_frames_dir = os.path.join(stage1_dir, "sampled_frames")
    manifest_path = os.path.join(stage1_dir, "frame_manifest.json")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    raw_path = os.path.join(stage1_dir, "stage1_raw_response.txt")
    sys_prompt_path = os.path.join(stage1_dir, "stage1_system_prompt.txt")
    user_prompt_path = os.path.join(stage1_dir, "stage1_user_prompt.txt")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    will_resume = not overwrite and _can_resume_stage1(draft_path, manifest_path)
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 1",
        overwrite=overwrite,
        allow_unfingerprinted_resume=allow_unfingerprinted_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage1] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        return video_out

    logger.info(
        f"[stage1] video_id={vid} start: overwrite={bool(overwrite)} "
        f"max_frames={int(sampling_cfg.max_frames)} src={os.path.abspath(video_path)}"
    )


    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "stage1": {"status": "running"},
        },
    )

    frames, dims = sample_video_to_frames(video_path, sampling_cfg)
    logger.info(f"[stage1] video_id={vid} sampled_frames={len(frames)} dims={dims}")
    save_sampled_frames_jpegs(frames, sampled_frames_dir)
    write_frame_manifest(frames, sampled_frames_dir, manifest_path)

    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")

    user_prompt = build_stage1_user_prompt(len(frames), dims)
    write_text(sys_prompt_path, SYSTEM_PROMPT_ANALYST)
    write_text(user_prompt_path, user_prompt)

    frames_content = build_api_content(
        frames,
        embed_index=False,
        include_manifest=False,
        include_frame_labels=False,
    )
    base_user_content = [{"type": "text", "text": user_prompt}] + frames_content
    system_msg = {"role": "system", "content": SYSTEM_PROMPT_ANALYST}

    last_content = ""
    last_errors: List[str] = []
    normalized: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    attempts = 0
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        logger.info(f"[stage1] video_id={vid} model_call attempt={attempt}/{max_retries}")
        if attempt == 1:
            user_content = base_user_content
        else:
            prefix = build_retry_prefix(last_errors, last_content)
            user_content = [{"type": "text", "text": prefix + user_prompt}] + frames_content

        messages = [system_msg, {"role": "user", "content": user_content}]
        content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
        stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        stage_usage["total_tokens"] += usage.get("total_tokens", 0)
        stage_usage["api_calls"] += 1
        last_content = content

        try:
            clean = extract_json_from_response(content)
            plan = json.loads(clean)
        except Exception as e:
            last_errors = [f"JSON parse error: {e}"]
            normalized = None
            continue

        raw_errors = _stage1_raw_schema_errors(plan)
        normalized, warnings = normalize_draft_plan(plan)
        step_count = len(normalized.get("steps", [])) if isinstance(normalized, dict) else 0
        if step_count and step_count != 1:
            warnings.append(f"Step count is {step_count} (required exactly 1).")
        if isinstance(normalized, dict):
            for st in normalized.get("steps", []):
                if not isinstance(st, dict):
                    continue
                sid = st.get("step_id")
                sg = str(st.get("step_goal", "")).strip()
                wc = len([w for w in sg.split() if w])
                if wc > 35:
                    warnings.append(f"step_id={sid} step_goal has {wc} words (preferred <= 35).")
        last_errors = raw_errors + _draft_hard_errors(normalized)
        if not last_errors:
            break
        normalized = None

    write_text(raw_path, last_content)
    if normalized is None:
        raise RuntimeError(f"Stage 1 failed after {attempts} attempts: " + " | ".join(last_errors[:10]))

    write_json(draft_path, normalized)
    step_count = len(normalized.get("steps", [])) if isinstance(normalized, dict) else 0
    logger.info(
        f"[stage1] video_id={vid} completed: steps={step_count} attempts={attempts} warnings={len(warnings)} "
        f"elapsed={format_duration(time.perf_counter() - t_start)}"
    )

    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "api_config": {
                "api_base_url": api_cfg.api_base_url,
                "model_provider_id": api_cfg.model_provider_id,
                "model_name": api_cfg.model_name,
                "max_tokens": int(api_cfg.max_tokens),
                "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
            },
            "sampling_config": {
                "max_frames": sampling_cfg.max_frames,
                "resize_dimension": sampling_cfg.resize_dimension,
                "jpeg_quality": sampling_cfg.jpeg_quality,
            },
            "stage1": {
                "status": "completed",
                "attempts": attempts,
                "manifest_path": os.path.relpath(manifest_path, video_out),
                "draft_plan_path": os.path.relpath(draft_path, video_out),
                "raw_response_path": os.path.relpath(raw_path, video_out),
                "system_prompt_path": os.path.relpath(sys_prompt_path, video_out),
                "user_prompt_path": os.path.relpath(user_prompt_path, video_out),
                "warnings": warnings,
                "token_usage": stage_usage,
            },
        },
    )

    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: generate draft plan (no critical_frames).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root for generated stage outputs.")

    add_api_cli_args(parser, include_no_embed_index=False)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unfingerprinted-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (outputs without schema fingerprints).",
    )
    args = parser.parse_args()

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        run_stage1_for_video(
            vp,
            args.output_root,
            api_cfg,
            sampling_cfg,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            allow_unfingerprinted_resume=args.allow_unfingerprinted_resume,
        )


if __name__ == "__main__":
    main()
