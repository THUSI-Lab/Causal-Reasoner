


from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from four_stage_common import (
    add_api_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_api_content,
    build_retry_prefix,
    can_open_video,
    call_chat_completion,
    collect_videos,
    cut_video_segment_ffmpeg,
    default_output_root,
    estimate_min_positive_delta_sec,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    initialize_api_client,
    guard_schema_fingerprint,
    logger,
    load_frames_from_manifest,
    read_json,
    sample_frames_around_timestamp,
    sanitize_filename,
    update_run_summary,
    validate_stage2_localization,
    video_id_from_path,
    write_json,
    write_text,
)
from four_stage_prompt_templates import (
    build_stage2_boundary_refinement_prompt,
    build_stage2_boundary_verification_prompt,
    build_stage2_user_prompt,
)

if TYPE_CHECKING:
    from four_stage_common import ApiConfig


def _adjust_end_sec_if_needed(
    *,
    step_id: int,
    start_sec: float,
    end_sec: float,
    min_delta_sec: float,
    next_step_start_sec: Optional[float],
    timestamps: List[float],
    end_index_1based: int,
) -> float:
    if end_sec > start_sec:
        return end_sec

    raw_start = float(start_sec)
    raw_end = float(end_sec)


    if next_step_start_sec is not None and float(next_step_start_sec) > raw_start:
        budget = float(next_step_start_sec) - raw_start
        eps = min(max(min_delta_sec, 0.05), budget * 0.9)
        adjusted = raw_start + eps
        logger.warning(
            f"[stage2] Adjusted non-positive duration for step_id={step_id}: "
            f"start={raw_start:.3f}, end={raw_end:.3f} -> end={adjusted:.3f} (budget to next step={budget:.3f})"
        )
        return adjusted


    for t in (timestamps or [])[max(0, int(end_index_1based) - 1) :]:
        tt = float(t)
        if tt > raw_start:
            logger.warning(
                f"[stage2] Adjusted non-positive duration for step_id={step_id}: "
                f"start={raw_start:.3f}, end={raw_end:.3f} -> end={tt:.3f} (next available timestamp)"
            )
            return tt


    adjusted = raw_start + max(min_delta_sec, 0.1)
    logger.warning(
        f"[stage2] Adjusted non-positive duration for step_id={step_id}: "
        f"start={raw_start:.3f}, end={raw_end:.3f} -> end={adjusted:.3f} (epsilon fallback)"
    )
    return adjusted


def _extract_step_goals_by_id(items: Any) -> Dict[int, str]:

    out: Dict[int, str] = {}
    if not isinstance(items, list):
        return out
    for obj in items:
        if not isinstance(obj, dict) or obj.get("step_id") is None:
            continue
        try:
            sid = int(obj.get("step_id"))
        except Exception:
            continue
        goal = str(obj.get("step_goal", "")).strip()
        if sid > 0 and goal:
            out[sid] = goal
    return out


def _normalize_independence_value(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    if s in {"yes", "no"}:
        return s
    return None


def _draft_has_required_independence_fields(draft: Any) -> bool:
    steps = draft.get("steps", []) if isinstance(draft, dict) else []
    if not isinstance(steps, list) or not steps:
        return False
    for st in steps:
        if not isinstance(st, dict):
            return False
        try:
            sid = int(st.get("step_id"))
        except Exception:
            return False
        if sid <= 1:
            continue
        if _normalize_independence_value(st.get("independence")) is None:
            return False
    return True


def _segments_have_required_independence_fields(segments: Any) -> bool:
    if not isinstance(segments, list) or not segments:
        return False
    for seg in segments:
        if not isinstance(seg, dict):
            return False
        try:
            sid = int(seg.get("step_id"))
        except Exception:
            return False
        if sid <= 1:
            continue
        if _normalize_independence_value(seg.get("independence")) is None:
            return False
    return True


def _can_resume_stage2(segments_path: str, draft_path: str, stage2_dir: str) -> bool:

    if not (os.path.exists(segments_path) and os.path.exists(draft_path)):
        return False
    try:
        segments_data = read_json(segments_path)
        segs = segments_data.get("segments", [])
        if not isinstance(segs, list) or not segs:
            return False
        draft = read_json(draft_path)
    except Exception:
        return False


    try:
        video_out = os.path.dirname(stage2_dir)
        manifest_path = os.path.join(video_out, "stage1", "frame_manifest.json")
        manifest = read_json(manifest_path)
        seg_num = int(segments_data.get("num_frames", -1))
        man_num = int(manifest.get("num_frames", -1))
        if seg_num <= 0 or man_num <= 0 or seg_num != man_num:
            return False
        frames = manifest.get("frames", [])
        if not isinstance(frames, list) or len(frames) != man_num:
            return False
        ts_list = [float(fr.get("timestamp_sec", 0.0)) if isinstance(fr, dict) else 0.0 for fr in frames]
    except Exception:
        return False

    expected = _extract_step_goals_by_id(draft.get("steps", []))
    got = _extract_step_goals_by_id(segs)
    if not expected or expected != got:
        return False
    if not _draft_has_required_independence_fields(draft):
        return False
    if not _segments_have_required_independence_fields(segs):
        return False


    try:
        seg_by_id = {int(seg.get("step_id")): seg for seg in segs if isinstance(seg, dict) and seg.get("step_id") is not None}
        loc_steps: List[Dict[str, int]] = []
        for sid in sorted(expected):
            seg = seg_by_id.get(int(sid))
            if not isinstance(seg, dict):
                return False
            entry: Dict[str, Any] = {
                "step_id": int(sid),
                "start_frame_index": int(seg.get("start_frame_index")),
                "end_frame_index": int(seg.get("end_frame_index")),
            }
            if int(sid) > 1:
                indep = _normalize_independence_value(seg.get("independence"))
                if indep is not None:
                    entry["independence"] = indep
            loc_steps.append(entry)
        ok, _errors, _by_id = validate_stage2_localization(
            draft, {"steps": loc_steps}, man_num, frame_timestamps=ts_list
        )
        if not ok:
            return False
    except Exception:
        return False

    for seg in segs:
        if not isinstance(seg, dict):
            return False
        rel = seg.get("clip_relpath")
        if not rel:
            return False
        clip_abs = os.path.join(stage2_dir, rel)
        if not os.path.exists(clip_abs) or os.path.getsize(clip_abs) <= 0:
            return False
        if not can_open_video(clip_abs):
            return False
    return True


def run_stage2_for_video(
    video_path: str,
    output_root: str,
    api_cfg: ApiConfig,
    ffmpeg_bin: str,
    overwrite: bool,
    max_retries: int,
    *,
    cut_mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
    allow_unfingerprinted_resume: bool = False,
) -> str:
    t_start = time.perf_counter()
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    manifest_path = os.path.join(stage1_dir, "frame_manifest.json")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    raw_path = os.path.join(stage2_dir, "stage2_raw_response.txt")
    loc_path = os.path.join(stage2_dir, "localization_raw.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    clips_dir = os.path.join(stage2_dir, "step_clips")
    sys_prompt_path = os.path.join(stage2_dir, "stage2_system_prompt.txt")
    user_prompt_path = os.path.join(stage2_dir, "stage2_user_prompt.txt")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    will_resume = not overwrite and _can_resume_stage2(segments_path, draft_path, stage2_dir)
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 2",
        overwrite=overwrite,
        allow_unfingerprinted_resume=allow_unfingerprinted_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage2] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        return video_out

    logger.info(
        f"[stage2] video_id={vid} start: overwrite={bool(overwrite)} cut_mode={cut_mode} "
        f"src={os.path.abspath(video_path)}"
    )


    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        )

    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Stage 1 draft not found: {draft_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Stage 1 manifest not found: {manifest_path}")

    draft_plan = read_json(draft_path)
    high_level_goal = str(draft_plan.get("high_level_goal", "")).strip()
    steps_for_count = draft_plan.get("steps", [])
    if not isinstance(steps_for_count, list) or not (1 <= len(steps_for_count) <= 9):
        raise RuntimeError(
            f"Draft step count must be within [1, 9] (got {len(steps_for_count) if isinstance(steps_for_count, list) else 'N/A'}). "
            "Re-run Stage 1 with a better prompt (or use --overwrite)."
        )

    steps = draft_plan.get("steps", [])
    ordered_steps = sorted([s for s in steps if isinstance(s, dict)], key=lambda x: int(x.get("step_id", 0)))




    if len(ordered_steps) == 1:
        st0 = ordered_steps[0]
        sid = int(st0.get("step_id", 0))
        goal = str(st0.get("step_goal", "")).strip()

        frames = load_frames_from_manifest(manifest_path)
        num_frames = len(frames)
        ts_list = [float(fr.get("timestamp_sec", 0.0)) for fr in frames]
        frames_meta = {int(e["frame_index_1based"]): e for e in read_json(manifest_path).get("frames", [])
                       if isinstance(e, dict) and "frame_index_1based" in e}

        start_sec = float(ts_list[0])
        end_sec = float(ts_list[-1]) + estimate_min_positive_delta_sec(ts_list)


        localization = {"steps": [{"step_id": sid, "start_frame_index": 1, "end_frame_index": num_frames + 1}]}

        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)
        write_json(loc_path, localization)


        slug = sanitize_filename(goal)
        clip_name = f"step{sid:02d}_{slug}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        abs_video_path = os.path.abspath(video_path)
        if os.path.exists(clip_path) or os.path.islink(clip_path):
            os.remove(clip_path)
        os.symlink(abs_video_path, clip_path)

        segments = [{
            "step_id": sid,
            "step_goal": goal,
            "start_frame_index": 1,
            "end_frame_index": num_frames + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_image_relpath": frames_meta.get(1, {}).get("image_relpath"),
            "end_image_relpath": frames_meta.get(num_frames, {}).get("image_relpath"),
            "clip_relpath": os.path.relpath(clip_path, stage2_dir),
        }]

        write_json(segments_path, {
            "source_video": abs_video_path,
            "video_id": vid,
            "num_frames": num_frames,
            "cut": {"mode": "symlink", "note": "source video is used as clip for the single step (no cutting)"},
            "segments": segments,
        })

        update_run_summary(run_summary_path, {
            "source_video": abs_video_path,
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "stage2": {
                "status": "completed",
                "localization_raw_path": os.path.relpath(loc_path, video_out),
                "segments_path": os.path.relpath(segments_path, video_out),
                "clips_dir": os.path.relpath(clips_dir, video_out),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0},
                "single_step_fast_path": True,
            },
        })

        logger.info(
            f"[stage2] video_id={vid} completed (single-step symlink): clips=1 "
            f"elapsed={format_duration(time.perf_counter() - t_start)}"
        )
        return video_out




    outline_lines: List[str] = []
    for st in ordered_steps:
        try:
            sid = int(st.get("step_id", 0))
        except Exception:
            continue
        goal = str(st.get("step_goal", "")).strip()
        if sid > 0 and goal:
            outline_lines.append(f"- Step {sid}: {goal}")
    draft_plan_outline = "\n".join(outline_lines)

    frames = load_frames_from_manifest(manifest_path)
    num_frames = len(frames)
    ts_list = [float(fr.get("timestamp_sec", 0.0)) for fr in frames]
    logger.info(f"[stage2] video_id={vid} draft_steps={len(ordered_steps)} stage1_frames={num_frames}")

    base_prompt = build_stage2_user_prompt(high_level_goal, draft_plan_outline, num_frames)
    system_text = "You are an expert video step temporal localization assistant. Return strict JSON only (no markdown, no extra text)."

    last_content = ""
    last_errors: List[str] = []
    localization: Optional[Dict[str, Any]] = None
    loc_by_id: Dict[int, Dict[str, int]] = {}
    localization_from_cache = False
    cached_loc_mtime: Optional[float] = None
    client = None
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}


    if not overwrite and os.path.exists(loc_path):
        try:
            loc_mtime = float(os.path.getmtime(loc_path))
            input_mtime = max(float(os.path.getmtime(draft_path)), float(os.path.getmtime(manifest_path)))
            if loc_mtime + 1e-6 < input_mtime:
                logger.warning(
                    f"[stage2] Cached localization is older than current Stage1 inputs; ignoring: {os.path.relpath(loc_path, video_out)}"
                )
            else:
                cached_loc = read_json(loc_path)
                ok, errors, by_id = validate_stage2_localization(
                    draft_plan, cached_loc, num_frames, frame_timestamps=ts_list
                )
                if ok:
                    localization = cached_loc
                    loc_by_id = by_id
                    localization_from_cache = True
                    cached_loc_mtime = loc_mtime
                    if not os.path.exists(sys_prompt_path):
                        write_text(sys_prompt_path, system_text)
                    if not os.path.exists(user_prompt_path):
                        write_text(user_prompt_path, base_prompt)
                    logger.info(f"[stage2] video_id={vid} reusing cached localization: {os.path.relpath(loc_path, video_out)}")
                else:
                    last_errors = errors
        except Exception as e:
            last_errors = [f"Failed to load cached localization: {e}"]

    if localization is None:


        if not overwrite and os.path.isdir(clips_dir):
            existing = [n for n in os.listdir(clips_dir) if n.lower().endswith(".mp4")]
            if existing:
                raise RuntimeError(
                    f"Found existing stage2 clips under {clips_dir} but no valid cached localization at {loc_path}. "
                    "To avoid mismatched clips/segments, delete stage2/step_clips or re-run with --overwrite."
                )

        client = initialize_api_client(api_cfg)
        if not client:
            raise SystemExit("Failed to initialize API client.")


        write_text(sys_prompt_path, system_text)
        write_text(user_prompt_path, base_prompt)

        frames_content = build_api_content(
            frames,
            api_cfg.embed_index_on_api_images,
            include_manifest=False,



            include_frame_labels=True,
        )
        base_user_content = [{"type": "text", "text": base_prompt}] + frames_content
        system_msg = {"role": "system", "content": system_text}

        for attempt in range(1, max_retries + 1):
            logger.info(f"[stage2] video_id={vid} localize attempt={attempt}/{max_retries}")
            if attempt == 1:
                user_content = base_user_content
            else:
                prefix = build_retry_prefix(last_errors, last_content)
                user_content = [{"type": "text", "text": prefix + base_prompt}] + frames_content

            messages = [system_msg, {"role": "user", "content": user_content}]
            content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 12000))
            stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            stage_usage["total_tokens"] += usage.get("total_tokens", 0)
            stage_usage["api_calls"] += 1
            last_content = content

            try:
                clean = extract_json_from_response(content)
                localization = json.loads(clean)
            except Exception as e:
                last_errors = [f"JSON parse error: {e}"]
                localization = None
                continue

            ok, errors, by_id = validate_stage2_localization(draft_plan, localization, num_frames, frame_timestamps=ts_list)
            if ok:
                loc_by_id = by_id
                break
            last_errors = errors
            localization = None

    if localization is None:
        if last_content:
            write_text(raw_path, last_content)
        raise RuntimeError(f"Stage 2 failed after {max_retries} attempts: " + " | ".join(last_errors[:10]))

    logger.info(f"[stage2] video_id={vid} localization OK (source={'cache' if localization_from_cache else 'model'})")

    if client is None:
        client = initialize_api_client(api_cfg)
        if not client:
            raise SystemExit("Failed to initialize API client.")




    if localization is not None and not localization_from_cache:
        verification_prompt = build_stage2_boundary_verification_prompt(
            high_level_goal,
            draft_plan_outline,
            num_frames,
            json.dumps(localization, indent=2),
        )

        verify_sys_prompt_path = os.path.join(stage2_dir, "stage2_verify_system_prompt.txt")
        verify_user_prompt_path = os.path.join(stage2_dir, "stage2_verify_user_prompt.txt")
        verify_raw_path = os.path.join(stage2_dir, "stage2_verify_raw_response.txt")

        verify_system_text = (
            "You are an expert video step boundary verifier. "
            "Return strict JSON only (no markdown, no extra text)."
        )
        write_text(verify_sys_prompt_path, verify_system_text)
        write_text(verify_user_prompt_path, verification_prompt)

        verify_user_content = [{"type": "text", "text": verification_prompt}] + frames_content
        verify_system_msg = {"role": "system", "content": verify_system_text}
        verify_messages = [verify_system_msg, {"role": "user", "content": verify_user_content}]

        logger.info(f"[stage2] video_id={vid} boundary verification pass starting")

        try:
            verify_content, verify_usage = call_chat_completion(
                client, api_cfg, verify_messages,
                max_tokens=min(api_cfg.max_tokens, 12000),
            )
            stage_usage["prompt_tokens"] += verify_usage.get("prompt_tokens", 0)
            stage_usage["completion_tokens"] += verify_usage.get("completion_tokens", 0)
            stage_usage["total_tokens"] += verify_usage.get("total_tokens", 0)
            stage_usage["api_calls"] += 1
            write_text(verify_raw_path, verify_content)

            verify_clean = extract_json_from_response(verify_content)
            verified_loc = json.loads(verify_clean)

            v_ok, v_errors, v_by_id = validate_stage2_localization(
                draft_plan, verified_loc, num_frames, frame_timestamps=ts_list,
            )

            if v_ok:
                changed = []
                for sid_v in sorted(loc_by_id):
                    orig = loc_by_id[sid_v]
                    veri = v_by_id.get(sid_v, {})
                    if (orig.get("start_frame_index") != veri.get("start_frame_index")
                            or orig.get("end_frame_index") != veri.get("end_frame_index")):
                        changed.append(
                            f"step_id={sid_v}: "
                            f"[{orig['start_frame_index']},{orig['end_frame_index']}) -> "
                            f"[{veri['start_frame_index']},{veri['end_frame_index']})"
                        )
                if changed:
                    logger.info(
                        f"[stage2] video_id={vid} verification adjusted {len(changed)} "
                        f"boundaries: " + "; ".join(changed)
                    )
                else:
                    logger.info(
                        f"[stage2] video_id={vid} verification confirmed all boundaries unchanged"
                    )
                localization = verified_loc


                for sid_v in loc_by_id:
                    if "independence" in loc_by_id[sid_v] and sid_v in v_by_id:
                        v_by_id[sid_v]["independence"] = loc_by_id[sid_v]["independence"]
                loc_by_id = v_by_id
            else:
                logger.warning(
                    f"[stage2] video_id={vid} verification output invalid, "
                    f"keeping original boundaries. Errors: {v_errors[:5]}"
                )
        except Exception as e:
            logger.warning(
                f"[stage2] video_id={vid} verification pass failed ({e}), keeping original boundaries"
            )





    if localization is not None and not localization_from_cache:
        ordered_sids = sorted(loc_by_id.keys())

        _step_goals: Dict[int, str] = {}
        for _st in ordered_steps:
            _sid_val = int(_st.get("step_id", 0))
            _step_goals[_sid_val] = str(_st.get("step_goal", "")).strip()

        for b_idx in range(len(ordered_sids) - 1):
            sid_i = ordered_sids[b_idx]
            sid_j = ordered_sids[b_idx + 1]
            boundary_frame = int(loc_by_id[sid_i]["end_frame_index"])


            if boundary_frame <= len(ts_list):
                boundary_sec = float(ts_list[boundary_frame - 1])
            else:
                boundary_sec = float(ts_list[-1])

            try:
                dense_frames, dense_ts = sample_frames_around_timestamp(
                    video_path, boundary_sec, window_sec=2.0, num_dense_frames=20,
                )
                if len(dense_frames) < 4:
                    continue                            

                goal_i = _step_goals.get(sid_i, "")
                goal_j = _step_goals.get(sid_j, "")
                refine_prompt = build_stage2_boundary_refinement_prompt(
                    goal_i, goal_j, len(dense_frames),
                    current_boundary_description=f"approximately frame {boundary_frame} in the global {num_frames}-frame pool",
                )

                dense_content = build_api_content(
                    dense_frames, True,
                    include_manifest=False,
                    include_frame_labels=True,
                    label_prefix="DenseFrame",
                )
                refine_messages = [
                    {"role": "system", "content": "You are an expert video step boundary specialist. Return strict JSON only."},
                    {"role": "user", "content": [{"type": "text", "text": refine_prompt}] + dense_content},
                ]

                refine_content, refine_usage = call_chat_completion(
                    client, api_cfg, refine_messages,
                    max_tokens=min(api_cfg.max_tokens, 2000),
                )
                stage_usage["prompt_tokens"] += refine_usage.get("prompt_tokens", 0)
                stage_usage["completion_tokens"] += refine_usage.get("completion_tokens", 0)
                stage_usage["total_tokens"] += refine_usage.get("total_tokens", 0)
                stage_usage["api_calls"] += 1

                refine_clean = extract_json_from_response(refine_content)
                refine_result = json.loads(refine_clean)
                refined_idx = int(refine_result.get("refined_boundary_frame_index", 0))
                if refined_idx < 1 or refined_idx > len(dense_ts):
                    logger.warning(
                        f"[stage2] boundary refinement for step {sid_i}/{sid_j}: "
                        f"invalid frame index {refined_idx}, skipping"
                    )
                    continue


                refined_ts = dense_ts[refined_idx - 1]

                new_boundary_frame = min(
                    range(1, num_frames + 2),
                    key=lambda f: abs(ts_list[min(f - 1, len(ts_list) - 1)] - refined_ts),
                )

                new_boundary_frame = max(2, min(new_boundary_frame, num_frames))


                if abs(new_boundary_frame - boundary_frame) > 8:
                    logger.info(
                        f"[stage2] boundary refinement for step {sid_i}/{sid_j}: "
                        f"proposed {new_boundary_frame} too far from {boundary_frame}, skipping"
                    )
                    continue

                if new_boundary_frame != boundary_frame:
                    loc_by_id[sid_i]["end_frame_index"] = new_boundary_frame
                    loc_by_id[sid_j]["start_frame_index"] = new_boundary_frame

                    for step_entry in (localization.get("steps", []) if isinstance(localization, dict) else localization):
                        if not isinstance(step_entry, dict):
                            continue
                        se_sid = int(step_entry.get("step_id", 0))
                        if se_sid == sid_i:
                            step_entry["end_frame_index"] = new_boundary_frame
                        elif se_sid == sid_j:
                            step_entry["start_frame_index"] = new_boundary_frame
                    logger.info(
                        f"[stage2] boundary refined: step {sid_i}/{sid_j} "
                        f"frame {boundary_frame} -> {new_boundary_frame} "
                        f"(confidence={refine_result.get('confidence', 'unknown')})"
                    )
                else:
                    logger.info(
                        f"[stage2] boundary refinement confirmed: step {sid_i}/{sid_j} "
                        f"frame {boundary_frame} unchanged"
                    )
            except Exception as e:
                logger.warning(
                    f"[stage2] boundary refinement failed for step {sid_i}/{sid_j}: {e}"
                )
                continue



    if last_content:
        write_text(raw_path, last_content)
        write_json(loc_path, localization)
    elif overwrite or not os.path.exists(loc_path):
        write_json(loc_path, localization)


    manifest = read_json(manifest_path)
    frames_meta = {int(e["frame_index_1based"]): e for e in manifest.get("frames", []) if isinstance(e, dict) and "frame_index_1based" in e}

    os.makedirs(clips_dir, exist_ok=True)

    segments: List[Dict[str, Any]] = []

    timestamps = ts_list
    min_delta_sec = estimate_min_positive_delta_sec(timestamps)

    bounds: List[Dict[str, Any]] = []
    for step in ordered_steps:
        sid = int(step.get("step_id", 0))
        goal = str(step.get("step_goal", "")).strip()
        seg_idx = loc_by_id.get(sid)
        if not seg_idx:
            raise RuntimeError(f"Missing localization for step_id={sid} after validation.")
        bounds.append(
            {
                "step_id": sid,
                "step_goal": goal,
                "start_frame_index": int(seg_idx["start_frame_index"]),
                "end_frame_index": int(seg_idx["end_frame_index"]),
            }
        )

    for i, b in enumerate(bounds):
        step_no = i + 1
        total_steps = len(bounds)
        sid = int(b["step_id"])
        goal = str(b["step_goal"])
        sidx = int(b["start_frame_index"])
        eidx = int(b["end_frame_index"])
        start_sec = float(timestamps[sidx - 1])
        if eidx == num_frames + 1:

            end_sec = float(timestamps[-1]) + float(min_delta_sec)
        else:



            boundary_ts = float(timestamps[eidx - 1])                                             
            prev_frame_ts = float(timestamps[eidx - 2])                                            



            inter_frame_gap = boundary_ts - prev_frame_ts
            end_sec = boundary_ts - inter_frame_gap * 0.5

            if end_sec <= prev_frame_ts:
                end_sec = prev_frame_ts + min(min_delta_sec, 0.05) * 0.5
        next_start_sec: Optional[float] = None
        if i + 1 < len(bounds):
            next_sidx = int(bounds[i + 1]["start_frame_index"])
            next_start_sec = float(timestamps[next_sidx - 1])
        end_sec = _adjust_end_sec_if_needed(
            step_id=sid,
            start_sec=start_sec,
            end_sec=end_sec,
            min_delta_sec=min_delta_sec,
            next_step_start_sec=next_start_sec,
            timestamps=timestamps,
            end_index_1based=eidx,
        )
        if not (end_sec > start_sec):
            raise RuntimeError(
                f"Invalid (non-positive) clip duration after adjustment for step_id={sid}: start_sec={start_sec}, end_sec={end_sec}"
            )

        slug = sanitize_filename(goal)
        clip_name = f"step{sid:02d}_{slug}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        goal_short = " ".join(str(goal).split())
        logger.info(
            f"[stage2] video_id={vid} clip {step_no}/{total_steps} step_id={sid}: "
            f"{start_sec:.2f}s..{end_sec:.2f}s goal='{goal_short}' -> {os.path.relpath(clip_path, video_out)}"
        )
        if (
            not overwrite
            and localization_from_cache
            and os.path.exists(clip_path)
            and os.path.getsize(clip_path) > 0
        ):
            need_recut = False
            if cached_loc_mtime is not None:
                try:
                    clip_mtime = float(os.path.getmtime(clip_path))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to verify existing clip freshness vs cached localization: {clip_path}: {e}"
                    ) from e

                if clip_mtime + 1.0 < float(cached_loc_mtime):
                    logger.warning(
                        f"[stage2] video_id={vid} existing clip is older than cached localization; re-cutting to match: "
                        f"{os.path.relpath(clip_path, video_out)}"
                    )
                    need_recut = True
            if not can_open_video(clip_path):
                logger.warning(
                    f"[stage2] video_id={vid} existing clip is unreadable; re-cutting: {os.path.relpath(clip_path, video_out)}"
                )
                need_recut = True
            if need_recut:
                cut_video_segment_ffmpeg(
                    ffmpeg_bin,
                    video_path,
                    start_sec,
                    end_sec,
                    clip_path,
                    overwrite=True,
                    mode=cut_mode,
                    seek_slop_sec=seek_slop_sec,
                    crf=crf,
                    preset=preset,
                    keep_audio=keep_audio,
                )
                if not can_open_video(clip_path):
                    raise RuntimeError(f"ffmpeg re-cut produced an unreadable clip: {clip_path}")
            else:
                logger.info(
                    f"[stage2] video_id={vid} reusing existing clip (overwrite disabled): {os.path.relpath(clip_path, video_out)}"
                )
        elif not overwrite and os.path.exists(clip_path):
            raise RuntimeError(
                f"Found an existing clip but overwrite is disabled: {clip_path}. "
                "This usually means a previous Stage-2 run was interrupted and you are now generating a NEW localization; "
                "to avoid mismatched clips/segments, delete stage2/step_clips or re-run with --overwrite."
            )
        else:
            cut_video_segment_ffmpeg(
                ffmpeg_bin,
                video_path,
                start_sec,
                end_sec,
                clip_path,
                overwrite=overwrite,
                mode=cut_mode,
                seek_slop_sec=seek_slop_sec,
                crf=crf,
                preset=preset,
                keep_audio=keep_audio,
            )
            if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
                raise RuntimeError(f"ffmpeg produced an empty clip: {clip_path}")
            if not can_open_video(clip_path):
                raise RuntimeError(f"ffmpeg produced an unreadable clip: {clip_path}")

        segments.append(
            {
                "step_id": sid,
                "step_goal": goal,
                "start_frame_index": sidx,
                "end_frame_index": eidx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_image_relpath": frames_meta.get(sidx, {}).get("image_relpath"),
                "end_image_relpath": frames_meta.get(min(eidx, num_frames), {}).get("image_relpath"),
                "clip_relpath": os.path.relpath(clip_path, stage2_dir),
            }
        )


    independence_by_id: Dict[int, str] = {}
    for sid_loc, seg_data in loc_by_id.items():
        if sid_loc > 1 and "independence" in seg_data:
            independence_by_id[sid_loc] = seg_data["independence"]

    for seg in segments:
        sid = int(seg.get("step_id", 0))
        if sid <= 1:
            seg.pop("independence", None)
            continue
        seg["independence"] = independence_by_id[sid]

    draft_steps_mut = draft_plan.get("steps", [])
    if not isinstance(draft_steps_mut, list):
        raise RuntimeError("Draft plan steps missing/invalid while injecting independence.")
    for st in draft_steps_mut:
        if not isinstance(st, dict):
            continue
        try:
            sid = int(st.get("step_id", 0))
        except Exception:
            continue
        if sid <= 1:
            st.pop("independence", None)
            continue
        st["independence"] = independence_by_id[sid]
    write_json(draft_path, draft_plan)

    write_json(
        segments_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "num_frames": num_frames,
            "cut": {
                "mode": cut_mode,
                "seek_slop_sec": float(seek_slop_sec),
                "crf": int(crf),
                "preset": preset,
                "keep_audio": bool(keep_audio),
                "ffmpeg_bin": ffmpeg_bin,
            },
            "segments": segments,
        },
    )

    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "stage2": {
                "status": "completed",
                "localization_raw_path": os.path.relpath(loc_path, video_out),
                "segments_path": os.path.relpath(segments_path, video_out),
                "clips_dir": os.path.relpath(clips_dir, video_out),
                "system_prompt_path": os.path.relpath(sys_prompt_path, video_out),
                "user_prompt_path": os.path.relpath(user_prompt_path, video_out),
                "api_config": {
                    "api_base_url": api_cfg.api_base_url,
                    "model_provider_id": api_cfg.model_provider_id,
                    "model_name": api_cfg.model_name,
                    "max_tokens": int(api_cfg.max_tokens),
                    "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                    "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                    "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
                },
                "token_usage": stage_usage,
            },
        },
    )

    logger.info(
        f"[stage2] video_id={vid} completed: clips={len(segments)} elapsed={format_duration(time.perf_counter() - t_start)}"
    )
    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: localize steps and cut per-step clips.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root for generated stage outputs.")

    add_api_cli_args(parser, include_no_embed_index=True)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--cut-mode", choices=["copy", "reencode"], default="reencode", help="ffmpeg cut mode (copy is fast but may snap to keyframes).")
    parser.add_argument("--seek-slop-sec", type=float, default=1.0, help="Re-encode mode: pre-seek slop (seconds) for hybrid accurate seek.")
    parser.add_argument("--crf", type=int, default=18, help="Re-encode mode: x264 CRF (lower=better quality).")
    parser.add_argument("--preset", default="veryfast", help="Re-encode mode: x264 preset.")
    parser.add_argument("--keep-audio", action="store_true", help="Keep/re-encode audio into the clip (default: drop audio).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unfingerprinted-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (outputs without schema fingerprints).",
    )
    args = parser.parse_args()

    api_cfg = api_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        run_stage2_for_video(
            vp,
            args.output_root,
            api_cfg,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            cut_mode=args.cut_mode,
            seek_slop_sec=args.seek_slop_sec,
            crf=args.crf,
            preset=args.preset,
            keep_audio=args.keep_audio,
            allow_unfingerprinted_resume=args.allow_unfingerprinted_resume,
        )


if __name__ == "__main__":
    main()
