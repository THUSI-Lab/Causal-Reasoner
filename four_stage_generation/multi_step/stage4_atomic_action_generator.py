




from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from four_stage_common import (
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_api_content,
    build_retry_prefix,

    call_chat_completion,
    collect_videos,
    cut_video_segment_ffmpeg,
    default_output_root,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    initialize_api_client,
    guard_schema_fingerprint,
    logger,
    read_json,
    sample_video_to_frames,
    sampling_config_from_args,
    sanitize_filename,
    save_sampled_frames_jpegs,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from four_stage_prompt_templates import (
    SYSTEM_PROMPT_ANALYST,
    build_stage4_user_prompt,
)

if TYPE_CHECKING:
    from four_stage_common import ApiConfig, SamplingConfig







def _merge_tiny_atomic_actions(
    actions: List[Dict[str, Any]],
    min_frame_span: int = 2,
) -> List[Dict[str, Any]]:

    if not actions:
        return actions


    merged = [dict(a) for a in actions]

    changed = True
    while changed:
        changed = False
        new_list: List[Dict[str, Any]] = []
        skip_next = False
        for idx in range(len(merged)):
            if skip_next:
                skip_next = False
                continue
            act = merged[idx]
            span = act["end_frame_index"] - act["start_frame_index"]
            if span >= min_frame_span:
                new_list.append(act)
                continue

            changed = True
            prev = new_list[-1] if new_list else None
            nxt = merged[idx + 1] if idx + 1 < len(merged) else None

            if prev is None and nxt is not None:

                nxt = dict(nxt)
                nxt["start_frame_index"] = act["start_frame_index"]
                merged[idx + 1] = nxt
            elif nxt is None and prev is not None:

                prev["end_frame_index"] = act["end_frame_index"]
            elif prev is not None and nxt is not None:
                prev_span = prev["end_frame_index"] - prev["start_frame_index"]
                nxt_span = nxt["end_frame_index"] - nxt["start_frame_index"]
                if prev_span >= nxt_span:

                    prev["end_frame_index"] = act["end_frame_index"]
                else:

                    nxt = dict(nxt)
                    nxt["start_frame_index"] = act["start_frame_index"]
                    merged[idx + 1] = nxt
            else:

                new_list.append(act)
        merged = new_list


    for idx, act in enumerate(merged):
        act["atomic_action_id"] = idx + 1

    return merged


def validate_stage4_atomic_actions(
    obj: Any,
    step_id: int,
    num_frames: int,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:

    errors: List[str] = []

    if not isinstance(obj, dict):
        return None, ["Response is not a JSON object."]


    raw_sid = obj.get("step_id")
    if raw_sid is None:
        errors.append("Missing 'step_id'.")
    else:
        try:
            got_sid = int(raw_sid)
        except (ValueError, TypeError):
            errors.append(f"'step_id' is not an integer: {raw_sid!r}")
            got_sid = None
        if got_sid is not None and got_sid != int(step_id):
            errors.append(f"'step_id' mismatch: expected {int(step_id)}, got {got_sid}.")


    actions = obj.get("atomic_actions")
    if not isinstance(actions, list) or len(actions) == 0:
        errors.append("'atomic_actions' must be a non-empty list.")
        return None, errors

    REQUIRED_KEYS = {
        "atomic_action_id", "start_frame_index", "end_frame_index",
        "actor", "action", "patient", "caption",
    }


    _FRAME_REF_RE = re.compile(
        r"(?:frame[_\s]*(?:index|#|num(?:ber)?)?[_\s]*[:=]?\s*\d)|"
        r"(?:\b(?:at|from|to|around|between)\s+\d+(?:\.\d+)?\s*(?:s|sec|seconds?|ms)\b)|"
        r"(?:timestamp\s*[:=]?\s*\d)",
        re.IGNORECASE,
    )

    normalized_actions: List[Dict[str, Any]] = []
    prev_end: Optional[int] = None

    for i, act in enumerate(actions):
        prefix = f"atomic_actions[{i}]"
        if not isinstance(act, dict):
            errors.append(f"{prefix}: not a JSON object.")
            continue


        missing = REQUIRED_KEYS - set(act.keys())
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}.")
            continue


        try:
            aid = int(act["atomic_action_id"])
        except (ValueError, TypeError):
            errors.append(f"{prefix}: 'atomic_action_id' is not an integer.")
            continue
        if aid != i + 1:
            errors.append(f"{prefix}: 'atomic_action_id' must be {i + 1} (sequential), got {aid}.")


        try:
            s = int(act["start_frame_index"])
            e = int(act["end_frame_index"])
        except (ValueError, TypeError):
            errors.append(f"{prefix}: start/end frame_index must be integers.")
            continue

        if not (1 <= s <= num_frames):
            errors.append(f"{prefix}: start_frame_index={s} out of range [1, {num_frames}].")
        if not (2 <= e <= num_frames + 1):
            errors.append(f"{prefix}: end_frame_index={e} out of range [2, {num_frames + 1}].")
        if s >= e:
            errors.append(f"{prefix}: start_frame_index ({s}) must be < end_frame_index ({e}).")


        if i == 0 and s != 1:
            errors.append(f"{prefix}: first atomic action must start at 1, got {s}.")
        if prev_end is not None and s != prev_end:
            errors.append(f"{prefix}: contiguity violation: expected start={prev_end}, got {s}.")
        prev_end = e


        for key in ("actor", "action", "patient", "caption"):
            val = act.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{prefix}: '{key}' must be a non-empty string.")


        for key in ("action", "caption", "patient"):
            val = str(act.get(key, "")).strip()
            if val and _FRAME_REF_RE.search(val):
                errors.append(f"{prefix}: '{key}' must not reference frame indices or timestamps.")


        raw_patient = str(act.get("patient", "")).strip()
        if raw_patient:
            raw_patient = raw_patient.lower()


        normalized_actions.append({
            "atomic_action_id": i + 1,
            "start_frame_index": s,
            "end_frame_index": e,
            "actor": str(act.get("actor", "")).strip(),
            "action": str(act.get("action", "")).strip(),
            "patient": raw_patient,
            "caption": str(act.get("caption", "")).strip(),
        })


    if normalized_actions:
        last_end = normalized_actions[-1]["end_frame_index"]
        if last_end != num_frames + 1:
            errors.append(
                f"Last atomic action end_frame_index must be {num_frames + 1} (full coverage), got {last_end}."
            )


    if len(normalized_actions) < 2 and not errors:
        errors.append(
            f"A step must decompose into at least 2 atomic actions (got {len(normalized_actions)}). "
            "Even simple steps have distinct phases (e.g., approach + execute)."
        )


    MAX_ATOMIC_ACTIONS = 25
    if len(normalized_actions) > MAX_ATOMIC_ACTIONS and not errors:
        errors.append(
            f"Too many atomic actions: got {len(normalized_actions)} (max {MAX_ATOMIC_ACTIONS}). "
            f"Each atomic action should be a COMPLETE functional operation (pick up, place, cut, open), "
            f"NOT a kinematic primitive. Merge reach+grasp+lift into one 'pick up' action, "
            f"merge lower+place+release into one 'place' action, and collapse repeated cycles."
        )

    if errors:
        return None, errors

    normalized = {
        "step_id": int(step_id),
        "atomic_actions": normalized_actions,
    }
    return normalized, []






def _can_resume_stage4_step(
    step_out_path: str,
    step_id: int,
    num_frames: int,
    atomic_clips_dir: str,
) -> Optional[Dict[str, Any]]:

    if not os.path.exists(step_out_path):
        return None
    try:
        data = read_json(step_out_path)
    except Exception:
        return None

    normalized, errs = validate_stage4_atomic_actions(data, step_id, num_frames)
    if normalized is None:
        return None


    for act in normalized.get("atomic_actions", []):
        aid = int(act["atomic_action_id"])
        slug = sanitize_filename(str(act.get("action", "")))
        clip_name = f"atomic_{aid:02d}_{slug}.mp4"
        clip_path = os.path.join(atomic_clips_dir, clip_name)
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
            return None

    return normalized


def _can_resume_stage4_final(
    final_path: str,
    segments_path: str,
    stage3_plan_path: str,
) -> bool:

    if not os.path.exists(final_path):
        return False
    try:
        data = read_json(final_path)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return False


    for st in steps:
        if not isinstance(st, dict):
            return False
        if st.get("step_id") is None or not st.get("atomic_actions"):
            return False
        try:
            sid = int(st.get("step_id"))
        except Exception:
            return False
        if sid > 1:
            independence = str(st.get("independence", "") or "").strip().lower()
            detail = st.get("detail_independence")
            if independence not in {"yes", "no"}:
                return False
            if not isinstance(detail, str):
                return False
            if independence == "yes" and not detail.strip():
                return False
            if independence == "no" and detail != "":
                return False
        acts = st["atomic_actions"]
        if not isinstance(acts, list) or not acts:
            return False


    if not os.path.exists(segments_path):
        return False
    if not os.path.exists(stage3_plan_path):
        return False


    try:
        plan = read_json(stage3_plan_path)
        plan_sids = set()
        for s in plan.get("steps", []):
            if isinstance(s, dict) and s.get("step_id") is not None:
                sid = int(s["step_id"])
                if sid > 1:
                    independence = str(s.get("independence", "") or "").strip().lower()
                    detail = s.get("detail_independence")
                    if independence not in {"yes", "no"}:
                        return False
                    if not isinstance(detail, str):
                        return False
                    if independence == "yes" and not detail.strip():
                        return False
                    if independence == "no" and detail != "":
                        return False
                plan_sids.add(sid)
        final_sids = set()
        for s in steps:
            if isinstance(s, dict) and s.get("step_id") is not None:
                final_sids.add(int(s["step_id"]))
        if plan_sids != final_sids:
            return False
    except Exception:
        return False

    return True






def run_stage4_for_video(
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    sampling_cfg: "SamplingConfig",
    overwrite: bool,
    max_retries: int,
    *,
    ffmpeg_bin: str = "ffmpeg",
    cut_mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
    allow_unfingerprinted_resume: bool = False,
) -> str:

    t_start = time.perf_counter()

    _max_frames = int(getattr(sampling_cfg, "max_frames", 0) or 0)
    if _max_frames <= 0 or _max_frames > 100:
        raise RuntimeError(
            "Stage 4 requires --max-frames between 1 and 100 (per-step clip frame pool). "
            f"Got max_frames={getattr(sampling_cfg, 'max_frames', None)}."
        )

    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)

    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    stage4_dir = os.path.join(video_out, "stage4")
    os.makedirs(stage4_dir, exist_ok=True)


    stage3_plan_path = os.path.join(stage3_dir, "causal_plan_with_keyframes.json")
    if not os.path.exists(stage3_plan_path):
        stage3_plan_path = os.path.join(video_out, "causal_plan_with_keyframes.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = os.path.join(stage4_dir, "atomic_plan_with_clips.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")


    if not os.path.exists(stage3_plan_path):
        raise FileNotFoundError(f"Stage 3 plan not found: {stage3_plan_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")


    will_resume = not overwrite and _can_resume_stage4_final(
        final_path, segments_path, stage3_plan_path,
    )






    _rs_for_fp: Dict[str, Any] = {}
    if os.path.exists(run_summary_path):
        try:
            _rs_for_fp = read_json(run_summary_path)
        except Exception:
            pass
    _stage4_ever_ran = isinstance(_rs_for_fp.get("stage4"), dict)
    _overwrite_for_fp = overwrite or (not _stage4_ever_ran)
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 4",
        overwrite=_overwrite_for_fp,
        allow_unfingerprinted_resume=allow_unfingerprinted_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage4] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")

        stage4_obj: Dict[str, Any] = {}
        if os.path.exists(run_summary_path):
            try:
                rs = read_json(run_summary_path)
            except Exception:
                rs = {}
            if isinstance(rs, dict) and isinstance(rs.get("stage4"), dict):
                stage4_obj = dict(rs.get("stage4") or {})
        stage4_obj = {"status": "completed"}
        stage4_obj["final_plan_path"] = os.path.relpath(final_path, video_out)
        update_run_summary(
            run_summary_path,
            {
                "source_video": os.path.abspath(video_path),
                "video_id": vid,
                "output_root": os.path.abspath(output_root),
                "schema_fingerprint": schema_fp,
                "stage4": stage4_obj,
            },
        )
        return video_out

    logger.info(
        f"[stage4] video_id={vid} start: overwrite={bool(overwrite)} max_frames={int(sampling_cfg.max_frames)} "
        f"src={os.path.abspath(video_path)}"
    )


    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        )


    stage3_plan = read_json(stage3_plan_path)
    high_level_goal = str(stage3_plan.get("high_level_goal", "")).strip()
    plan_steps = stage3_plan.get("steps", [])
    if not isinstance(plan_steps, list) or not plan_steps:
        raise RuntimeError("Stage 3 plan has no steps.")

    segments_data = read_json(segments_path)
    segs = segments_data.get("segments", [])
    if not isinstance(segs, list) or not segs:
        raise RuntimeError("Stage 2 segments missing/empty.")
    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if isinstance(seg, dict) and seg.get("step_id") is not None:
            seg_by_id[int(seg["step_id"])] = seg


    ordered_steps = sorted(
        [s for s in plan_steps if isinstance(s, dict)],
        key=lambda x: int(x.get("step_id", 0)),
    )
    total_steps = len(ordered_steps)
    logger.info(f"[stage4] video_id={vid} steps={total_steps}")

    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")

    final_steps: List[Dict[str, Any]] = []
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}




    _all_patients: List[str] = []
    for _s in ordered_steps:
        cc = _s.get("causal_chain", {})
        if cc.get("patient"):
            _all_patients.append(str(cc["patient"]).strip())

    _seen_patients: set = set()
    _unique_patients: List[str] = []
    for _p in _all_patients:
        if _p and _p not in _seen_patients:
            _seen_patients.add(_p)
            _unique_patients.append(_p)
    global_entity_registry = ""
    if _unique_patients:
        global_entity_registry = "\n".join(f"  - \"{p}\"" for p in _unique_patients)

    for i_step, step in enumerate(ordered_steps, start=1):
        sid = int(step.get("step_id", 0))
        step_goal = str(step.get("step_goal", "")).strip()
        step_goal_short = " ".join(step_goal.split())

        if sid not in seg_by_id:
            raise RuntimeError(f"Missing Stage 2 segment for step_id={sid}")
        seg = seg_by_id[sid]
        try:
            clip_start_sec = float(seg.get("start_sec", 0.0))
            clip_end_sec = float(seg.get("end_sec", 0.0))
        except Exception:
            clip_start_sec = 0.0
            clip_end_sec = 0.0
        clip_rel = seg.get("clip_relpath")
        if not clip_rel:
            raise RuntimeError(f"Missing clip_relpath for step_id={sid}")
        clip_path = os.path.join(stage2_dir, clip_rel)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip not found: {clip_path}")


        slug = sanitize_filename(step_goal)
        step_dir_name = f"step{sid:02d}_{slug}"
        step_out_dir = os.path.join(stage4_dir, step_dir_name)
        atomic_clips_dir = os.path.join(step_out_dir, "atomic_clips")
        step_out_path = os.path.join(step_out_dir, "atomic_actions.json")
        manifest_path = os.path.join(step_out_dir, "frame_manifest.json")
        raw_path = os.path.join(step_out_dir, "stage4_raw_response.txt")
        sys_prompt_path = os.path.join(step_out_dir, "stage4_system_prompt.txt")
        user_prompt_path = os.path.join(step_out_dir, "stage4_user_prompt.txt")


        if not overwrite:

            cached_num_frames = int(sampling_cfg.max_frames)

            if os.path.exists(manifest_path):
                try:
                    cached_manifest = read_json(manifest_path)
                    cached_num_frames = int(cached_manifest.get("num_frames", 0) or 0)
                    if cached_num_frames <= 0:
                        cached_num_frames = len(cached_manifest.get("frames", []) or []) or int(sampling_cfg.max_frames)
                except Exception:
                    cached_num_frames = int(sampling_cfg.max_frames)

            cached = _can_resume_stage4_step(
                step_out_path, sid, cached_num_frames, atomic_clips_dir,
            )
            if cached is not None:
                logger.info(
                    f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
                    f"reuse cached atomic_actions goal='{step_goal_short}'"
                )

                final_entry = _build_final_step_entry(
                    cached, step, seg, stage2_dir, step_dir_name, atomic_clips_dir,
                    manifest_path, clip_start_sec, clip_end_sec,
                )
                final_steps.append(final_entry)
                continue

        step_started = time.perf_counter()
        logger.info(
            f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
            f"goal='{step_goal_short}' clip={os.path.relpath(clip_path, video_out)} "
            f"({clip_start_sec:.2f}s..{clip_end_sec:.2f}s)"
        )

        os.makedirs(step_out_dir, exist_ok=True)
        os.makedirs(atomic_clips_dir, exist_ok=True)


        sampled_frames, _ = sample_video_to_frames(clip_path, sampling_cfg)
        num_frames = len(sampled_frames)


        for fr in sampled_frames:
            try:
                fr["timestamp_sec"] = float(fr.get("timestamp_sec", 0.0)) + clip_start_sec
            except Exception:
                fr["timestamp_sec"] = clip_start_sec




        clip_duration = clip_end_sec - clip_start_sec
        local_timestamps: List[float] = []
        for fr in sampled_frames:
            local_t = float(fr.get("timestamp_sec", 0.0)) - clip_start_sec

            local_t = max(0.0, min(local_t, clip_duration))
            local_timestamps.append(local_t)


        sampled_frames_dir = os.path.join(step_out_dir, "sampled_frames")
        save_sampled_frames_jpegs(sampled_frames, sampled_frames_dir)
        write_frame_manifest(sampled_frames, sampled_frames_dir, manifest_path)


        step_annotation_json = json.dumps(step, ensure_ascii=False)

        next_step_goal = ""
        if i_step < total_steps:
            next_step_goal = str(ordered_steps[i_step].get("step_goal", "")).strip()
        base_prompt = build_stage4_user_prompt(
            high_level_goal=high_level_goal,
            step_goal=step_goal,
            step_annotation_json=step_annotation_json,
            num_frames=num_frames,
            next_step_goal=next_step_goal,
            global_entity_registry=global_entity_registry,
        )

        write_text(sys_prompt_path, SYSTEM_PROMPT_ANALYST)
        write_text(user_prompt_path, base_prompt)

        frames_content = build_api_content(
            sampled_frames,
            api_cfg.embed_index_on_api_images,
            include_manifest=False,
            include_frame_labels=True,
        )
        base_user_content = [{"type": "text", "text": base_prompt}] + frames_content
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_ANALYST}


        last_content = ""
        last_errors: List[str] = []
        normalized_result: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            logger.info(
                f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid} "
                f"model_call attempt={attempt}/{max_retries}"
            )
            if attempt == 1:
                user_content = base_user_content
            else:
                prefix = build_retry_prefix(last_errors, last_content)
                user_content = [{"type": "text", "text": prefix + base_prompt}] + frames_content

            messages = [system_msg, {"role": "user", "content": user_content}]
            content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
            stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            stage_usage["total_tokens"] += usage.get("total_tokens", 0)
            stage_usage["api_calls"] += 1
            last_content = content

            try:
                clean = extract_json_from_response(content)
                obj = json.loads(clean)
            except Exception as e:
                last_errors = [f"JSON parse error: {e}"]
                continue

            normalized_result, errs = validate_stage4_atomic_actions(obj, sid, num_frames)
            if normalized_result is not None:

                old_count = len(normalized_result["atomic_actions"])
                normalized_result["atomic_actions"] = _merge_tiny_atomic_actions(
                    normalized_result["atomic_actions"],
                )
                new_count = len(normalized_result["atomic_actions"])
                if new_count < old_count:
                    logger.info(
                        f"[stage4] video_id={vid} step_id={sid}: "
                        f"auto-merged {old_count - new_count} tiny AA(s) "
                        f"({old_count} -> {new_count})"
                    )

                if new_count < 2:
                    normalized_result = None
                    errs = [
                        f"After auto-merging tiny actions, only {new_count} atomic action(s) remain "
                        f"(minimum is 2). Produce at least 2 actions with sufficient frame span."
                    ]
            if normalized_result is not None:
                break
            last_errors = errs

        if normalized_result is None:
            write_text(raw_path, last_content)
            raise RuntimeError(
                f"Stage 4 failed for step_id={sid} after {max_retries} attempts: "
                + " | ".join(last_errors[:10])
            )

        write_text(raw_path, last_content)
        write_json(step_out_path, normalized_result)


        MIN_CLIP_DURATION = 0.04                                    
        for act in normalized_result["atomic_actions"]:
            aid = int(act["atomic_action_id"])
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            action_slug = sanitize_filename(str(act.get("action", "")))
            clip_name = f"atomic_{aid:02d}_{action_slug}.mp4"
            dst_path = os.path.join(atomic_clips_dir, clip_name)


            local_start = local_timestamps[s_idx - 1]                      
            if e_idx - 1 < len(local_timestamps):
                local_end = local_timestamps[e_idx - 1]
            else:

                local_end = clip_duration


            duration = local_end - local_start
            if duration < MIN_CLIP_DURATION:

                local_end = min(local_start + MIN_CLIP_DURATION, clip_duration)
                if local_end - local_start < MIN_CLIP_DURATION:

                    local_start = max(0.0, local_end - MIN_CLIP_DURATION)
                logger.warning(
                    f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                    f"adjusted short duration ({duration:.3f}s) -> {local_end - local_start:.3f}s "
                    f"(local_start={local_start:.3f} local_end={local_end:.3f})"
                )

            logger.info(
                f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                f"cutting {local_start:.2f}s..{local_end:.2f}s (clip-local) -> {clip_name}"
            )

            cut_video_segment_ffmpeg(
                ffmpeg_bin,
                clip_path,
                local_start,
                local_end,
                dst_path,
                overwrite=True,
                mode=cut_mode,
                seek_slop_sec=seek_slop_sec,
                crf=crf,
                preset=preset,
                keep_audio=keep_audio,
            )

            if not os.path.exists(dst_path) or os.path.getsize(dst_path) <= 0:
                logger.warning(
                    f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                    f"ffmpeg produced an empty clip: {dst_path}"
                )


            act["clip_relpath"] = os.path.join(step_dir_name, "atomic_clips", clip_name)


        orig_timestamps = [float(fr.get("timestamp_sec", 0.0)) for fr in sampled_frames]



        clamped_end_sec = min(clip_end_sec, orig_timestamps[-1]) if orig_timestamps else clip_end_sec
        for act in normalized_result["atomic_actions"]:
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            act["start_sec"] = orig_timestamps[s_idx - 1]
            if e_idx - 1 < len(orig_timestamps):
                act["end_sec"] = orig_timestamps[e_idx - 1]
            else:
                act["end_sec"] = clamped_end_sec


        write_json(step_out_path, normalized_result)


        final_entry = _build_final_step_entry(
            normalized_result, step, seg, stage2_dir, step_dir_name, atomic_clips_dir,
            manifest_path, clip_start_sec, clip_end_sec,
        )
        final_steps.append(final_entry)

        logger.info(
            f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid} done: "
            f"atomic_actions={len(normalized_result['atomic_actions'])} "
            f"elapsed={format_duration(time.perf_counter() - step_started)}"
        )


    final_plan: Dict[str, Any] = {
        "high_level_goal": high_level_goal,
        "video_id": vid,
        "source_video": os.path.abspath(video_path),
        "steps": final_steps,
    }
    write_json(final_path, final_plan)


    stage3_step_by_id: Dict[int, Dict[str, Any]] = {
        int(s["step_id"]): s for s in plan_steps if isinstance(s, dict) and s.get("step_id") is not None
    }
    merged_steps: List[Dict[str, Any]] = []
    for s4_step in final_steps:
        sid = int(s4_step["step_id"])
        s3_step = stage3_step_by_id.get(sid, {})


        s4_clip = s4_step.get("clip_relpath", "")
        if s4_clip.startswith("../"):
            s4_clip = s4_clip[len("../"):]


        merged_atoms: List[Dict[str, Any]] = []
        for act in s4_step.get("atomic_actions", []):
            a = dict(act)                
            if a.get("clip_relpath") and not a["clip_relpath"].startswith("stage4/"):
                a["clip_relpath"] = os.path.join("stage4", a["clip_relpath"])
            merged_atoms.append(a)

        merged_step: Dict[str, Any] = {
            "step_id": sid,
            "step_goal": s3_step.get("step_goal", s4_step.get("step_goal", "")),
            "rationale": s3_step.get("rationale", ""),
            "causal_chain": s3_step.get("causal_chain", {}),
            "counterfactual_challenge_question": s3_step.get("counterfactual_challenge_question", ""),
            "expected_challenge_outcome": s3_step.get("expected_challenge_outcome", ""),
            "failure_reflecting": s3_step.get("failure_reflecting", {}),
            "critical_frames": s3_step.get("critical_frames", []),
            "clip_relpath": s4_clip,
            "atomic_actions": merged_atoms,
        }
        if sid > 1:
            merged_step["independence"] = s3_step.get("independence", s4_step.get("independence", ""))
            merged_step["detail_independence"] = s3_step.get(
                "detail_independence",
                s4_step.get("detail_independence", ""),
            )
        merged_steps.append(merged_step)

    final_merged: Dict[str, Any] = {
        "high_level_goal": high_level_goal,
        "video_id": vid,
        "source_video": os.path.abspath(video_path),
        "steps": merged_steps,
    }
    final_merged_path = os.path.join(video_out, "final_plan.json")

    from four_stage_common import strip_underscores_from_values
    final_merged = strip_underscores_from_values(final_merged)
    write_json(final_merged_path, final_merged)
    logger.info(f"[stage4] video_id={vid} final_plan.json written: {final_merged_path}")


    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "stage4": {
                "status": "completed",
                "token_usage": stage_usage,
                "final_plan_path": os.path.relpath(final_path, video_out),
                "merged_final_plan_path": "final_plan.json",
                "frame_index_note": (
                    "In this four-stage pipeline, atomic_actions[*].start_frame_index and end_frame_index are "
                    f"1-based on EACH STEP CLIP's {int(sampling_cfg.max_frames)}-frame pool (same as stage3); "
                    "see each step folder's frame_manifest.json."
                ),
                "api_config": {
                    "api_base_url": api_cfg.api_base_url,
                    "model_provider_id": api_cfg.model_provider_id,
                    "model_name": api_cfg.model_name,
                    "max_tokens": int(api_cfg.max_tokens),
                    "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                    "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                    "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
                },
            },
        },
    )

    logger.info(
        f"[stage4] video_id={vid} completed: steps={len(final_steps)} "
        f"elapsed={format_duration(time.perf_counter() - t_start)}"
    )
    return video_out


def _build_final_step_entry(
    normalized_result: Dict[str, Any],
    step: Dict[str, Any],
    seg: Dict[str, Any],
    stage2_dir: str,
    step_dir_name: str,
    atomic_clips_dir: str,
    manifest_path: str,
    clip_start_sec: float,
    clip_end_sec: float,
) -> Dict[str, Any]:

    sid = int(normalized_result["step_id"])
    step_goal = str(step.get("step_goal", "")).strip()
    clip_rel = seg.get("clip_relpath", "")


    clip_relpath_from_stage4 = os.path.join("..", "stage2", clip_rel) if clip_rel else ""


    enriched_actions: List[Dict[str, Any]] = []
    actions = normalized_result.get("atomic_actions", [])


    orig_timestamps: Optional[List[float]] = None
    needs_timestamps = actions and ("start_sec" not in actions[0])
    needs_clip_relpath = actions and ("clip_relpath" not in actions[0])

    if needs_timestamps and os.path.exists(manifest_path):
        try:
            manifest = read_json(manifest_path)
            frames = manifest.get("frames", [])
            if isinstance(frames, list) and frames:
                orig_timestamps = [
                    float(fr.get("timestamp_sec", 0.0)) if isinstance(fr, dict) else 0.0
                    for fr in frames
                ]
        except Exception:
            pass


    clamped_end_sec = (
        min(clip_end_sec, orig_timestamps[-1]) if orig_timestamps else clip_end_sec
    )

    for act in actions:
        entry = dict(act)                


        if needs_timestamps and orig_timestamps:
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            entry["start_sec"] = orig_timestamps[s_idx - 1] if s_idx - 1 < len(orig_timestamps) else clip_start_sec
            if e_idx - 1 < len(orig_timestamps):
                entry["end_sec"] = orig_timestamps[e_idx - 1]
            else:
                entry["end_sec"] = clamped_end_sec


        if needs_clip_relpath:
            aid = int(act["atomic_action_id"])
            action_slug = sanitize_filename(str(act.get("action", "")))
            clip_name = f"atomic_{aid:02d}_{action_slug}.mp4"
            entry["clip_relpath"] = os.path.join(step_dir_name, "atomic_clips", clip_name)

        enriched_actions.append(entry)

    return {
        "step_id": sid,
        "step_goal": step_goal,
        "clip_relpath": clip_relpath_from_stage4,
        "atomic_actions": enriched_actions,
        **(
            {
                "independence": step.get("independence", ""),
                "detail_independence": step.get("detail_independence", ""),
            }
            if sid > 1
            else {}
        ),
    }






def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: decompose each step into fine-grained atomic actions and cut atomic clips."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help="Output root for generated four-stage outputs.",
    )

    add_api_cli_args(parser, include_no_embed_index=True)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--cut-mode",
        choices=["copy", "reencode"],
        default="reencode",
        help="ffmpeg cut mode (copy is fast but may snap to keyframes).",
    )
    parser.add_argument(
        "--seek-slop-sec",
        type=float,
        default=1.0,
        help="Re-encode mode: pre-seek slop (seconds) for hybrid accurate seek.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Re-encode mode: x264 CRF (lower=better quality).",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="Re-encode mode: x264 preset.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep/re-encode audio into the clip (default: drop audio).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unfingerprinted-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (outputs without schema fingerprints).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Don't stop batch processing on single video failure.",
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

    succeeded = 0
    failed = 0
    for vp in videos:
        try:
            run_stage4_for_video(
                vp,
                args.output_root,
                api_cfg,
                sampling_cfg,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
                ffmpeg_bin=args.ffmpeg_bin,
                cut_mode=args.cut_mode,
                seek_slop_sec=args.seek_slop_sec,
                crf=args.crf,
                preset=args.preset,
                keep_audio=args.keep_audio,
                allow_unfingerprinted_resume=args.allow_unfingerprinted_resume,
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.error(f"[stage4] Failed for video {vp}: {e}")
            if not args.continue_on_error:
                raise

    if len(videos) > 1:
        logger.info(f"[stage4] Batch complete: {succeeded} succeeded, {failed} failed out of {len(videos)} videos.")


if __name__ == "__main__":
    main()
