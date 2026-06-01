


from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from four_stage_common import (
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_api_content,
    build_retry_prefix,
    can_open_video,
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
    load_frames_from_manifest,
    normalize_high_level_goal_text,
    normalize_stage3_step_output,
    read_json,
    sample_video_to_frames,
    sampling_config_from_args,
    stage3_step_folder_basename,
    save_keyframe_images_from_manifest,
    save_sampled_frames_jpegs,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from four_stage_prompt_templates import (
    SYSTEM_PROMPT_ANALYST,
    build_stage3_hlg_and_detail_independence_prompt,
    build_stage3_keyframe_alignment_user_prompt,
    build_stage3_user_prompt,
)

if TYPE_CHECKING:
    from four_stage_common import ApiConfig, SamplingConfig


def _normalize_independence_value(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    if s in {"yes", "no"}:
        return s
    return None


def _apply_dependency_fields(
    step_obj: Dict[str, Any],
    *,
    step_id: int,
    independence: Optional[str],
    detail_independence: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(step_obj)
    if int(step_id) <= 1:
        out.pop("independence", None)
        out.pop("detail_independence", None)
        return out
    if independence is None:
        raise RuntimeError(f"Missing independence for step_id={int(step_id)}.")
    out["independence"] = independence
    if detail_independence is not None:
        out["detail_independence"] = str(detail_independence).strip()
    elif independence == "no":
        out["detail_independence"] = ""
    else:
        out.pop("detail_independence", None)
    return out


def _step_has_required_dependency_fields(step_obj: Any) -> bool:
    if not isinstance(step_obj, dict):
        return False
    try:
        sid = int(step_obj.get("step_id"))
    except Exception:
        return False
    if sid <= 1:
        return "independence" not in step_obj and "detail_independence" not in step_obj
    independence = _normalize_independence_value(step_obj.get("independence"))
    if independence is None:
        return False
    detail = step_obj.get("detail_independence")
    if not isinstance(detail, str):
        return False
    if independence == "yes":
        return bool(detail.strip())
    return detail == ""


def _validate_stage3_detail_independence_result(
    obj: Any,
    steps: List[Dict[str, Any]],
) -> tuple[Optional[Dict[int, str]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return None, ["detail_independence output is not a JSON object."]
    items = obj.get("steps")
    if not isinstance(items, list):
        return None, ["detail_independence output must contain a 'steps' list."]

    expected_ids = [int(st.get("step_id")) for st in steps if isinstance(st, dict) and int(st.get("step_id", 0)) > 1]
    independence_by_id = {
        int(st.get("step_id")): _normalize_independence_value(st.get("independence"))
        for st in steps
        if isinstance(st, dict) and int(st.get("step_id", 0)) > 1
    }

    got: Dict[int, str] = {}
    for i, item in enumerate(items):
        prefix = f"steps[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} is not an object.")
            continue
        extra = sorted(set(item.keys()) - {"step_id", "detail_independence"})
        if extra:
            errors.append(f"{prefix} contains extra keys: {extra}")
        try:
            sid = int(item.get("step_id"))
        except Exception:
            errors.append(f"{prefix}.step_id is missing/invalid.")
            continue
        if sid <= 1:
            errors.append(f"{prefix}.step_id must be greater than 1.")
            continue
        detail = str(item.get("detail_independence", "") or "").strip()
        if sid in got:
            errors.append(f"Duplicate detail_independence entry for step_id={sid}.")
            continue
        independence = independence_by_id.get(sid)
        if independence == "yes" and not detail:
            errors.append(f"{prefix}.detail_independence must be non-empty when independence is 'yes'.")
            continue
        if independence == "no" and detail:
            errors.append(f"{prefix}.detail_independence must be empty when independence is 'no'.")
            continue
        got[sid] = detail

    if sorted(got.keys()) != sorted(expected_ids):
        errors.append(
            f"detail_independence output step_ids mismatch: expected {expected_ids}, got {sorted(got.keys())}."
        )
    if errors:
        return None, errors
    return got, []


def _auto_recut_clip_if_needed(
    *,
    video_id: str,
    src_video_path: str,
    clip_path: str,
    clip_start_sec: float,
    clip_end_sec: float,
    cut_cfg: Dict[str, Any],
    force: bool = False,
) -> bool:

    if not force and can_open_video(clip_path):
        return False

    ffmpeg_bin = str(cut_cfg.get("ffmpeg_bin") or "ffmpeg")
    mode = str(cut_cfg.get("mode") or "reencode")
    seek_slop_sec = float(cut_cfg.get("seek_slop_sec", 1.0) or 1.0)
    crf = int(cut_cfg.get("crf", 18) or 18)
    preset = str(cut_cfg.get("preset") or "veryfast")
    keep_audio = bool(cut_cfg.get("keep_audio", False))

    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"[stage3] video_id={video_id} cannot auto-recut unreadable clip (ffmpeg not found): {ffmpeg_bin}. "
            "Delete stage2/step_clips and re-run Stage 2 with a valid --ffmpeg-bin."
        )

    logger.warning(
        f"[stage3] video_id={video_id} clip is unreadable; auto-recutting via ffmpeg: {os.path.abspath(clip_path)}"
    )
    cut_video_segment_ffmpeg(
        ffmpeg_bin,
        src_video_path,
        clip_start_sec,
        clip_end_sec,
        clip_path,
        overwrite=True,
        mode=mode,
        seek_slop_sec=seek_slop_sec,
        crf=crf,
        preset=preset,
        keep_audio=keep_audio,
    )
    if not can_open_video(clip_path):
        raise RuntimeError(f"[stage3] video_id={video_id} auto-recut produced an unreadable clip: {clip_path}")
    return True


def _candidate_step_folders(stage3_dir: str, step_id: int) -> List[str]:
    prefix = f"{int(step_id):02d}_"
    try:
        names = os.listdir(stage3_dir)
    except Exception:
        return []
    out: List[str] = []
    for name in names:
        if not name.startswith(prefix):
            continue
        p = os.path.join(stage3_dir, name)
        if os.path.isdir(p):
            out.append(p)
    return sorted(out)


def _find_step_folder_for_segment(
    stage3_dir: str,
    step_id: int,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
) -> Optional[str]:

    matches: List[str] = []
    for cand in _candidate_step_folders(stage3_dir, step_id):
        meta_path = os.path.join(cand, "step_meta.json")
        if _step_meta_matches_segment(
            meta_path,
            expected_clip_abs,
            expected_clip_start_sec,
            expected_clip_end_sec,
        ):
            matches.append(cand)
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"[stage3] step_id={int(step_id)} has multiple matching step folders; using '{os.path.basename(matches[0])}': "
            + ", ".join(os.path.basename(m) for m in matches)
        )
    return sorted(matches)[0]


def _select_working_step_folder(
    stage3_dir: str,
    step_id: int,
    draft_step_goal: str,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
) -> str:

    matched = _find_step_folder_for_segment(
        stage3_dir,
        step_id,
        expected_clip_abs,
        expected_clip_start_sec,
        expected_clip_end_sec,
    )
    if matched:
        return matched



    cands = _candidate_step_folders(stage3_dir, step_id)
    if len(cands) == 1:
        return cands[0]
    if cands:
        logger.warning(
            f"[stage3] step_id={int(step_id)} has multiple step folders but none match the current segment; "
            f"using '{os.path.basename(cands[0])}': " + ", ".join(os.path.basename(p) for p in cands)
        )
        return cands[0]

    return os.path.join(stage3_dir, stage3_step_folder_basename(step_id, draft_step_goal))


def _final_step_folder(stage3_dir: str, step_id: int, final_step_goal: str) -> str:
    return os.path.join(stage3_dir, stage3_step_folder_basename(step_id, final_step_goal))


def _rename_step_folder_to_final(step_folder: str, *, stage3_dir: str, step_id: int, final_step_goal: str) -> str:

    src = os.path.abspath(step_folder)
    dst = os.path.abspath(_final_step_folder(stage3_dir, step_id, final_step_goal))
    if src == dst:
        return dst
    if not os.path.isdir(src):
        raise RuntimeError(f"Step folder not found for rename (step_id={int(step_id)}): {src}")

    if os.path.exists(dst):
        if not os.path.isdir(dst):
            raise RuntimeError(f"Cannot rename step folder to existing non-directory path: {dst}")


        conflict_base = f"conflict_{os.path.basename(dst)}_{int(time.time())}"
        conflict_path = os.path.join(stage3_dir, conflict_base)
        j = 0
        while os.path.exists(conflict_path):
            j += 1
            conflict_path = os.path.join(stage3_dir, f"{conflict_base}_{j}")
        logger.warning(
            f"[stage3] step_id={int(step_id)} destination step folder exists; moving it aside: "
            f"dst='{os.path.basename(dst)}' -> '{os.path.basename(conflict_path)}'"
        )
        os.rename(dst, conflict_path)

    os.rename(src, dst)
    return dst


def _ensure_step_folders_use_final_names(
    *,
    video_out: str,
    stage3_dir: str,
    final_plan: Dict[str, Any],
    seg_by_id: Dict[int, Dict[str, Any]],
) -> None:

    steps = final_plan.get("steps", [])
    if not isinstance(steps, list):
        return
    stage2_dir = os.path.join(video_out, "stage2")
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            continue
        try:
            sid = int(st.get("step_id"))
        except Exception:
            continue
        final_goal = str(st.get("step_goal", "")).strip()
        if sid <= 0 or not final_goal:
            continue
        seg = seg_by_id.get(sid) or {}
        clip_rel = seg.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            continue
        clip_abs = os.path.join(stage2_dir, clip_rel)
        try:
            clip_start_sec = float(seg.get("start_sec"))
            clip_end_sec = float(seg.get("end_sec"))
        except Exception:
            continue

        step_folder = _select_working_step_folder(
            stage3_dir,
            sid,
            draft_step_goal=final_goal,
            expected_clip_abs=clip_abs,
            expected_clip_start_sec=clip_start_sec,
            expected_clip_end_sec=clip_end_sec,
        )
        new_folder = _rename_step_folder_to_final(step_folder, stage3_dir=stage3_dir, step_id=sid, final_step_goal=final_goal)


        meta_path = os.path.join(new_folder, "step_meta.json")
        if os.path.exists(meta_path):
            try:
                meta = read_json(meta_path)
            except Exception:
                meta = None
            if isinstance(meta, dict):
                if str(meta.get("step_goal", "")).strip() != final_goal:
                    meta["step_goal"] = final_goal
                    write_json(meta_path, meta)


def _step_meta_matches_segment(
    step_meta_path: str,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
    *,
    tol_sec: float = 1e-3,
) -> bool:
    if not os.path.exists(step_meta_path):
        return False
    try:
        meta = read_json(step_meta_path)
    except Exception:
        return False

    rel = meta.get("clip_path")
    if not isinstance(rel, str) or not rel:
        return False
    step_folder = os.path.dirname(step_meta_path)
    meta_clip_abs = os.path.abspath(os.path.join(step_folder, rel))
    if os.path.abspath(expected_clip_abs) != meta_clip_abs:
        return False

    try:
        s = float(meta.get("clip_start_sec"))
        e = float(meta.get("clip_end_sec"))
    except Exception:
        return False
    if abs(s - float(expected_clip_start_sec)) > float(tol_sec):
        return False
    if abs(e - float(expected_clip_end_sec)) > float(tol_sec):
        return False
    return True


def _load_manifest_frame_timestamps(manifest_path: str) -> Optional[List[float]]:
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return None
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return None
    out: List[float] = []
    for fr in frames:
        if not isinstance(fr, dict):
            out.append(0.0)
            continue
        try:
            out.append(float(fr.get("timestamp_sec", 0.0)))
        except Exception:
            out.append(0.0)
    return out


def _expected_keyframe_image_paths(
    manifest_path: str,
    frame_indices_1based: List[int],
    step_folder: str,
) -> Dict[int, str]:
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return {}

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx1 = int(entry.get("frame_index_1based"))
        except Exception:
            continue
        by_idx[idx1] = entry

    out: Dict[int, str] = {}
    for idx1 in frame_indices_1based:
        entry = by_idx.get(int(idx1))
        if not entry:
            continue
        try:
            ts = float(entry.get("timestamp_sec", 0.0))
        except Exception:
            ts = 0.0
        name = f"frame_{int(idx1):03d}_ts_{ts:.2f}s.jpg"
        out[int(idx1)] = os.path.abspath(os.path.join(step_folder, name))
    return out


def _mean_abs_diff_b64_jpg(b64_a: str, b64_b: str) -> Optional[float]:

    try:
        import base64

        import cv2
        import numpy as np
    except Exception:
        return None

    try:
        a = np.frombuffer(base64.b64decode(b64_a), dtype=np.uint8)
        b = np.frombuffer(base64.b64decode(b64_b), dtype=np.uint8)
        img_a = cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imdecode(b, cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            return None
        img_a = cv2.resize(img_a, (64, 64), interpolation=cv2.INTER_AREA)
        img_b = cv2.resize(img_b, (64, 64), interpolation=cv2.INTER_AREA)
        diff = np.mean(np.abs(img_a.astype(np.float32) - img_b.astype(np.float32)))
        return float(diff)
    except Exception:
        return None


def _keyframe_similarity_error(sampled_frames: List[Dict[str, Any]], frame_indices_1based: List[int]) -> Optional[str]:

    if len(frame_indices_1based) != 2:
        return None
    try:
        i1 = int(frame_indices_1based[0])
        i2 = int(frame_indices_1based[1])
    except Exception:
        return None
    if not (1 <= i1 <= len(sampled_frames) and 1 <= i2 <= len(sampled_frames)):
        return None
    b64_a = sampled_frames[i1 - 1].get("base64")
    b64_b = sampled_frames[i2 - 1].get("base64")
    if not (isinstance(b64_a, str) and isinstance(b64_b, str) and b64_a and b64_b):
        return None


    if b64_a == b64_b:
        return (
            "Selected keyframe images are identical (duplicate frame content). "
            "Pick two distinct frames with clear visual/causal progression within the step."
        )


    mad = _mean_abs_diff_b64_jpg(b64_a, b64_b)
    if mad is not None and mad < 0.5:
        return (
            f"Selected keyframe images are near-identical (mean_abs_diff={mad:.2f}). "
            "Pick two frames that show clearer visual/causal progression within the step."
        )
    return None


def _unique_frame_indices_by_base64(sampled_frames: List[Dict[str, Any]]) -> List[int]:

    seen: set[str] = set()
    uniq: List[int] = []
    for i, fr in enumerate(sampled_frames, start=1):
        b64 = fr.get("base64")
        if not (isinstance(b64, str) and b64):
            continue
        if b64 in seen:
            continue
        seen.add(b64)
        uniq.append(i)
    return uniq


def _autofix_keyframes_if_possible(
    sampled_frames: List[Dict[str, Any]],
    chosen_indices_1based: List[int],
) -> Optional[List[int]]:

    if len(chosen_indices_1based) != 2:
        return None
    try:
        i1 = int(chosen_indices_1based[0])
        i2 = int(chosen_indices_1based[1])
    except Exception:
        return None
    if not (1 <= i1 < i2 <= len(sampled_frames)):
        return None

    uniq = _unique_frame_indices_by_base64(sampled_frames)
    if len(uniq) < 2:
        return None


    return [int(uniq[0]), int(uniq[-1])]


def _can_resume_stage3_final_plan(
    final_path: str,
    draft_path: str,
    segments_path: str,
    video_out: str,
    stage3_dir: str,
    *,
    max_frames_fallback: int,
) -> bool:

    if not os.path.exists(final_path):
        return False

    try:
        data = read_json(final_path)
    except Exception:
        return False

    high_level_goal = data.get("high_level_goal")
    steps = data.get("steps", [])
    if not (isinstance(high_level_goal, str) and high_level_goal.strip() and isinstance(steps, list) and steps):
        return False


    expected_by_id: Optional[Dict[int, str]] = None
    if os.path.exists(draft_path):
        try:
            draft = read_json(draft_path)
        except Exception:
            return False
        expected_by_id = {}
        for st in draft.get("steps", []):
            if not isinstance(st, dict) or st.get("step_id") is None:
                continue
            try:
                sid = int(st.get("step_id"))
            except Exception:
                continue
            goal = str(st.get("step_goal", "")).strip()
            if sid > 0 and goal:
                expected_by_id[sid] = goal
        if not expected_by_id:
            return False

    got_by_id: Dict[int, str] = {}
    step_by_id: Dict[int, Dict[str, Any]] = {}
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            return False
        try:
            sid = int(st.get("step_id"))
        except Exception:
            return False
        goal = str(st.get("step_goal", "")).strip()
        if sid <= 0 or not goal:
            return False
        if sid in got_by_id:
            return False
        got_by_id[sid] = goal
        step_by_id[sid] = st


    if expected_by_id is not None and set(expected_by_id.keys()) != set(got_by_id.keys()):
        return False



    if not os.path.exists(segments_path):
        return False
    try:
        seg_data = read_json(segments_path)
    except Exception:
        return False
    segs = seg_data.get("segments", [])
    if not isinstance(segs, list) or not segs:
        return False
    stage2_dir = os.path.join(video_out, "stage2")
    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if not isinstance(seg, dict) or seg.get("step_id") is None:
            continue
        try:
            sid = int(seg.get("step_id"))
        except Exception:
            continue
        seg_by_id[sid] = seg
    if not seg_by_id:
        return False



    for sid, goal in got_by_id.items():
        if sid not in seg_by_id:
            return False
        seg = seg_by_id[sid]
        try:
            clip_start_sec = float(seg.get("start_sec"))
            clip_end_sec = float(seg.get("end_sec"))
        except Exception:
            return False
        clip_rel = seg.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            return False
        clip_abs = os.path.join(stage2_dir, clip_rel)
        if not os.path.exists(clip_abs):
            return False


        desired = _final_step_folder(stage3_dir, sid, goal)
        step_folder: Optional[str] = None
        if os.path.isdir(desired):
            desired_meta_path = os.path.join(desired, "step_meta.json")
            if _step_meta_matches_segment(desired_meta_path, clip_abs, clip_start_sec, clip_end_sec):
                step_folder = desired
        if not step_folder:
            step_folder = _find_step_folder_for_segment(stage3_dir, sid, clip_abs, clip_start_sec, clip_end_sec)
        if not step_folder:
            return False

        step_meta_path = os.path.join(step_folder, "step_meta.json")
        if not _step_meta_matches_segment(step_meta_path, clip_abs, clip_start_sec, clip_end_sec):
            return False

        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        if not os.path.exists(manifest_path):
            return False
        try:
            manifest = read_json(manifest_path)
            num_frames = int(manifest.get("num_frames", 0) or 0)
            if num_frames <= 0:
                num_frames = len(manifest.get("frames", []) or []) or int(max_frames_fallback)
        except Exception:
            return False
        frame_timestamps = _load_manifest_frame_timestamps(manifest_path)

        step_out_path = os.path.join(step_folder, "step_final.json")
        if not os.path.exists(step_out_path):
            return False
        try:
            step_file = read_json(step_out_path)
        except Exception:
            return False

        normalized_file, errs_file = normalize_stage3_step_output(
            step_file, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_file is None:
            return False
        if not _step_has_required_dependency_fields(normalized_file):
            return False

        step_obj = step_by_id.get(sid)
        if not isinstance(step_obj, dict):
            return False
        normalized_final, errs_final = normalize_stage3_step_output(
            step_obj, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_final is None:
            return False
        if not _step_has_required_dependency_fields(normalized_final):
            return False
        if normalized_final != normalized_file:
            return False

        chosen = [int(cf["frame_index"]) for cf in normalized_file.get("critical_frames", [])]
        expected_paths = _expected_keyframe_image_paths(manifest_path, chosen, step_folder)
        if len(expected_paths) != len(chosen):
            return False
        if any(not os.path.exists(p) for p in expected_paths.values()):
            return False

    return True


def _load_valid_cached_step_final(
    step_out_path: str,
    manifest_path: str,
    step_meta_path: str,
    step_id: int,
    step_goal: str,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
    *,
    max_frames_fallback: int,
) -> Optional[Dict[str, Any]]:

    if not os.path.exists(step_out_path):
        return None
    if not _step_meta_matches_segment(step_meta_path, expected_clip_abs, expected_clip_start_sec, expected_clip_end_sec):
        return None
    try:
        if not os.path.exists(manifest_path):
            return None
        manifest = read_json(manifest_path)
        num_frames = int(manifest.get("num_frames", 0) or 0)
        if num_frames <= 0:
            num_frames = len(manifest.get("frames", []) or []) or int(max_frames_fallback)
        frame_timestamps = _load_manifest_frame_timestamps(manifest_path)
        existing = read_json(step_out_path)
    except Exception:
        return None
    if not isinstance(existing, dict):
        return None

    normalized, errs = normalize_stage3_step_output(
        existing, step_id, step_goal, num_frames, frame_timestamps=frame_timestamps
    )
    if normalized is None:
        return None

    step_folder = os.path.dirname(step_out_path)
    chosen = [int(cf["frame_index"]) for cf in normalized.get("critical_frames", [])]
    expected_paths = _expected_keyframe_image_paths(manifest_path, chosen, step_folder)
    if len(expected_paths) != len(chosen):
        return None
    if any(not os.path.exists(p) for p in expected_paths.values()):
        return None

    if existing != normalized:
        write_json(step_out_path, normalized)
    return normalized


def run_stage3_for_video(
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
            "Stage 3 requires --max-frames=100 (per-step clip frame pool) to keep keyframe index semantics consistent. "
            f"Got max_frames={getattr(sampling_cfg, 'max_frames', None)}."
        )
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    os.makedirs(stage3_dir, exist_ok=True)
    stage1_manifest_path = os.path.join(stage1_dir, "frame_manifest.json")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = os.path.join(stage3_dir, "causal_plan_with_keyframes.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Stage 1 draft not found: {draft_path}")
    if not os.path.exists(stage1_manifest_path):
        raise FileNotFoundError(f"Stage 1 manifest not found: {stage1_manifest_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")
    will_resume = not overwrite and _can_resume_stage3_final_plan(
        final_path,
        draft_path,
        segments_path,
        video_out,
        stage3_dir,
        max_frames_fallback=sampling_cfg.max_frames,
    )
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 3",
        overwrite=overwrite,
        allow_unfingerprinted_resume=allow_unfingerprinted_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage3] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        try:
            final_plan = read_json(final_path)
            segs = read_json(segments_path).get("segments", [])
            seg_by_id = {int(s.get("step_id")): s for s in segs if isinstance(s, dict) and s.get("step_id") is not None}
            _ensure_step_folders_use_final_names(video_out=video_out, stage3_dir=stage3_dir, final_plan=final_plan, seg_by_id=seg_by_id)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure final step folder names for video_id={vid}: {e}") from e



        stage3_obj: Dict[str, Any] = {}
        if os.path.exists(run_summary_path):
            try:
                rs = read_json(run_summary_path)
            except Exception:
                rs = {}
            if isinstance(rs, dict) and isinstance(rs.get("stage3"), dict):
                stage3_obj = dict(rs.get("stage3") or {})
        stage3_obj = {"status": "completed"}
        stage3_obj["final_plan_path"] = os.path.relpath(final_path, video_out)
        stage3_obj.setdefault(
            "frame_index_note",
            (
                "In this four-stage pipeline, critical_frames[*].frame_index is 1-based on EACH STEP CLIP's "
                f"{int(sampling_cfg.max_frames)}-frame pool; see each step folder's frame_manifest.json."
            ),
        )
        update_run_summary(
            run_summary_path,
            {
                "source_video": os.path.abspath(video_path),
                "video_id": vid,
                "output_root": os.path.abspath(output_root),
                "schema_fingerprint": schema_fp,
                "stage3": stage3_obj,
            },
        )
        return video_out

    logger.info(
        f"[stage3] video_id={vid} start: overwrite={bool(overwrite)} max_frames={int(sampling_cfg.max_frames)} "
        f"src={os.path.abspath(video_path)}"
    )

    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

    draft = read_json(draft_path)
    high_level_goal = str(draft.get("high_level_goal", "")).strip()
    stage1_frames = load_frames_from_manifest(stage1_manifest_path)
    draft_steps = draft.get("steps", [])
    if not isinstance(draft_steps, list) or not draft_steps:
        raise RuntimeError("Draft plan has no steps.")
    if not (3 <= len(draft_steps) <= 9):
        raise RuntimeError(
            f"Draft step count must be within [3, 9] for the four-stage pipeline (got {len(draft_steps)}). "
            "Re-run Stage 1 with a better prompt (or use --overwrite)."
        )

    segments_data = read_json(segments_path)
    segs = segments_data.get("segments", [])
    if not isinstance(segs, list) or not segs:
        raise RuntimeError("Stage 2 segments missing/empty.")
    cut_cfg = segments_data.get("cut", {})
    if not isinstance(cut_cfg, dict):
        cut_cfg = {}
    seg_by_id = {int(s.get("step_id")): s for s in segs if isinstance(s, dict) and s.get("step_id") is not None}

    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")

    final_steps: List[Dict[str, Any]] = []
    step_out_paths_by_id: Dict[int, str] = {}
    ordered_steps = sorted([s for s in draft_steps if isinstance(s, dict)], key=lambda x: int(x.get("step_id", 0)))
    total_steps = len(ordered_steps)
    logger.info(f"[stage3] video_id={vid} steps={total_steps} (sampling per-clip)")
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

    for i_step, step in enumerate(ordered_steps, start=1):
        sid = int(step.get("step_id", 0))
        goal = str(step.get("step_goal", "")).strip()
        goal_short = " ".join(goal.split())
        if sid not in seg_by_id:
            raise RuntimeError(f"Missing Stage 2 segment for step_id={sid}")
        seg = seg_by_id[sid]
        step_independence = _normalize_independence_value(step.get("independence"))
        if step_independence is None:
            step_independence = _normalize_independence_value(seg.get("independence"))
        if sid > 1 and step_independence is None:
            raise RuntimeError(f"Missing independence for step_id={sid} in Stage 2 outputs.")
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

        step_folder = _select_working_step_folder(
            stage3_dir,
            sid,
            draft_step_goal=goal,
            expected_clip_abs=clip_path,
            expected_clip_start_sec=clip_start_sec,
            expected_clip_end_sec=clip_end_sec,
        )
        os.makedirs(step_folder, exist_ok=True)
        sampled_frames_dir = os.path.join(step_folder, "sampled_frames")
        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        raw_path = os.path.join(step_folder, "stage3_raw_response.txt")
        step_out_path = os.path.join(step_folder, "step_final.json")
        step_meta_path = os.path.join(step_folder, "step_meta.json")
        step_out_paths_by_id[sid] = step_out_path



        if not overwrite:
            cached = _load_valid_cached_step_final(
                step_out_path,
                manifest_path,
                step_meta_path,
                sid,
                goal,
                expected_clip_abs=clip_path,
                expected_clip_start_sec=clip_start_sec,
                expected_clip_end_sec=clip_end_sec,
                max_frames_fallback=sampling_cfg.max_frames,
            )
            if cached is not None:
                logger.info(
                    f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
                    f"reuse cached step_final goal='{goal_short}'"
                )
                cached = _apply_dependency_fields(cached, step_id=sid, independence=step_independence)
                write_json(step_out_path, cached)
                final_steps.append(cached)
                continue

        step_started = time.perf_counter()
        logger.info(
            f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
            f"goal='{goal_short}' clip={os.path.relpath(clip_path, video_out)} ({clip_start_sec:.2f}s..{clip_end_sec:.2f}s)"
        )


        _auto_recut_clip_if_needed(
            video_id=vid,
            src_video_path=video_path,
            clip_path=clip_path,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            cut_cfg=cut_cfg,
        )
        try:
            sampled_frames, _ = sample_video_to_frames(clip_path, sampling_cfg)
        except RuntimeError as e:
            msg = str(e)
            if "Cannot open video" in msg or "invalid metadata" in msg:
                _auto_recut_clip_if_needed(
                    video_id=vid,
                    src_video_path=video_path,
                    clip_path=clip_path,
                    clip_start_sec=clip_start_sec,
                    clip_end_sec=clip_end_sec,
                    cut_cfg=cut_cfg,
                    force=True,
                )
                sampled_frames, _ = sample_video_to_frames(clip_path, sampling_cfg)
            else:
                raise


        for fr in sampled_frames:
            try:
                fr["timestamp_sec"] = float(fr.get("timestamp_sec", 0.0)) + clip_start_sec
            except Exception:
                fr["timestamp_sec"] = clip_start_sec
        frame_timestamps = [float(fr.get("timestamp_sec", 0.0)) for fr in sampled_frames]
        save_sampled_frames_jpegs(sampled_frames, sampled_frames_dir)
        write_frame_manifest(sampled_frames, sampled_frames_dir, manifest_path)

        draft_step_json = json.dumps(step, ensure_ascii=False)
        base_prompt = build_stage3_user_prompt(high_level_goal, draft_plan_outline, draft_step_json, len(sampled_frames))
        write_text(os.path.join(step_folder, "stage3_system_prompt.txt"), SYSTEM_PROMPT_ANALYST)
        write_text(os.path.join(step_folder, "stage3_user_prompt.txt"), base_prompt)
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
        normalized_step: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            logger.info(
                f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid} model_call attempt={attempt}/{max_retries}"
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

            normalized_step, errs = normalize_stage3_step_output(
                obj, sid, goal, len(sampled_frames), frame_timestamps=frame_timestamps
            )
            if normalized_step is not None:
                chosen = [int(cf["frame_index"]) for cf in normalized_step.get("critical_frames", [])]
                sim_err = _keyframe_similarity_error(sampled_frames, chosen)
                if sim_err:
                    uniq = _unique_frame_indices_by_base64(sampled_frames)
                    if len(uniq) < 2:
                        logger.warning(
                            f"[stage3] video_id={vid} step_id={sid}: frame pool has <2 unique images "
                            f"(unique={len(uniq)}/{len(sampled_frames)}); allowing duplicate/near-identical keyframes."
                        )
                        break

                    fixed = _autofix_keyframes_if_possible(sampled_frames, chosen)
                    if fixed and fixed != chosen:
                        logger.warning(
                            f"[stage3] video_id={vid} step_id={sid}: auto-fix keyframes due to similarity check: "
                            f"{chosen} -> {fixed} ({sim_err})"
                        )
                        normalized_step["critical_frames"][0]["frame_index"] = int(fixed[0])
                        normalized_step["critical_frames"][1]["frame_index"] = int(fixed[1])
                        break

                    logger.warning(
                        f"[stage3] video_id={vid} step_id={sid}: could not auto-fix keyframes; "
                        f"continuing with model-selected indices {chosen}. ({sim_err})"
                    )
                    break
                break
            last_errors = errs

        if normalized_step is None:
            write_text(raw_path, last_content)
            raise RuntimeError(f"Stage 3 failed for step_id={sid}: " + " | ".join(last_errors[:10]))

        write_text(raw_path, last_content)


        chosen = [int(cf["frame_index"]) for cf in normalized_step["critical_frames"]]
        kf_prompt = build_stage3_keyframe_alignment_user_prompt(
            step_id=sid,
            step_goal=str(normalized_step.get("step_goal", "")).strip(),
            critical_frames_json=json.dumps(normalized_step.get("critical_frames", []), ensure_ascii=False),
            num_frames=len(sampled_frames),
            frame_indices_1based=chosen,
        )
        kf_frames_content: List[Dict[str, Any]] = []
        for idx1 in chosen:
            if not (1 <= int(idx1) <= len(sampled_frames)):
                raise RuntimeError(f"Invalid keyframe index selected: {idx1} (num_frames={len(sampled_frames)})")
            b64 = sampled_frames[int(idx1) - 1].get("base64")
            if not isinstance(b64, str) or not b64:
                raise RuntimeError(f"Missing base64 for sampled frame {idx1}.")
            kf_frames_content.append(
                {"type": "text", "text": f"Keyframe image (locked): frame_index={int(idx1)} (1-based over full step clip)"}
            )
            kf_frames_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        kf_raw_path = os.path.join(step_folder, "stage3_keyframes_raw_response.txt")
        kf_user_prompt_path = os.path.join(step_folder, "stage3_keyframes_user_prompt.txt")
        write_text(kf_user_prompt_path, kf_prompt)

        last_kf_content = ""
        last_kf_errors: List[str] = []
        aligned_critical_frames: Optional[List[Dict[str, Any]]] = None
        for attempt in range(1, max_retries + 1):
            logger.info(
                f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid} keyframe_align attempt={attempt}/{max_retries}"
            )
            if attempt == 1:
                user_content = [{"type": "text", "text": kf_prompt}] + kf_frames_content
            else:
                prefix = build_retry_prefix(last_kf_errors, last_kf_content)
                user_content = [{"type": "text", "text": prefix + kf_prompt}] + kf_frames_content
            messages = [system_msg, {"role": "user", "content": user_content}]
            content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
            stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            stage_usage["total_tokens"] += usage.get("total_tokens", 0)
            stage_usage["api_calls"] += 1
            last_kf_content = content
            try:
                clean = extract_json_from_response(content)
                obj = json.loads(clean)
            except Exception as e:
                last_kf_errors = [f"JSON parse error: {e}"]
                continue
            if not isinstance(obj, dict):
                last_kf_errors = ["Keyframe alignment output is not an object."]
                continue

            cf_candidate = obj.get("critical_frames")
            merged = {**normalized_step, "critical_frames": cf_candidate}
            normalized_aligned, errs = normalize_stage3_step_output(
                merged, sid, goal, len(sampled_frames), frame_timestamps=frame_timestamps
            )
            if normalized_aligned is None:
                last_kf_errors = errs
                continue
            got_chosen = [int(cf["frame_index"]) for cf in normalized_aligned.get("critical_frames", [])]
            if got_chosen != chosen:
                last_kf_errors = [
                    "Keyframe alignment must keep critical_frames[*].frame_index locked to the originally selected indices: "
                    f"expected {chosen}, got {got_chosen}"
                ]
                continue
            aligned_critical_frames = list(normalized_aligned.get("critical_frames", []))
            break

        write_text(kf_raw_path, last_kf_content)
        if aligned_critical_frames is None:
            raise RuntimeError(f"Stage 3 keyframe alignment failed for step_id={sid}: " + " | ".join(last_kf_errors[:10]))


        normalized_step["critical_frames"] = aligned_critical_frames


        chosen = [int(cf["frame_index"]) for cf in normalized_step["critical_frames"]]
        save_keyframe_images_from_manifest(manifest_path, chosen, output_dir=step_folder)

        final_step_goal = str(normalized_step.get("step_goal", "")).strip() or goal
        normalized_step = _apply_dependency_fields(
            normalized_step,
            step_id=sid,
            independence=step_independence,
        )
        write_json(step_out_path, normalized_step)
        write_json(
            step_meta_path,
            {
                "step_id": sid,
                "step_goal": final_step_goal,
                "clip_path": os.path.relpath(clip_path, step_folder),
                "clip_start_sec": clip_start_sec,
                "clip_end_sec": clip_end_sec,
                "num_frames": len(sampled_frames),
                "manifest_path": os.path.relpath(manifest_path, step_folder),
            },
        )


        step_folder = _rename_step_folder_to_final(
            step_folder,
            stage3_dir=stage3_dir,
            step_id=sid,
            final_step_goal=final_step_goal,
        )
        step_out_paths_by_id[sid] = os.path.join(step_folder, "step_final.json")

        logger.info(
            f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid} done in "
            f"{format_duration(time.perf_counter() - step_started)}"
        )
        final_steps.append(normalized_step)

    refined_outline_lines: List[str] = []
    for st in final_steps:
        if not isinstance(st, dict):
            continue
        try:
            sid = int(st.get("step_id", 0))
        except Exception:
            continue
        sg = str(st.get("step_goal", "")).strip()
        if sid > 0 and sg:
            refined_outline_lines.append(f"- Step {sid}: {sg}")
    refined_plan_outline = "\n".join(refined_outline_lines) if refined_outline_lines else draft_plan_outline


    detail_steps_payload: List[Dict[str, Any]] = []
    for st in final_steps:
        if not isinstance(st, dict):
            continue
        payload: Dict[str, Any] = {
            "step_id": int(st.get("step_id", 0)),
            "step_goal": str(st.get("step_goal", "")).strip(),
            "causal_chain": st.get("causal_chain", {}),
        }
        if payload["step_id"] > 1:
            payload["independence"] = str(st.get("independence", "")).strip().lower()
        detail_steps_payload.append(payload)

    hlg_detail_prompt = build_stage3_hlg_and_detail_independence_prompt(
        draft_high_level_goal=high_level_goal,
        draft_plan_outline=draft_plan_outline,
        refined_plan_outline=refined_plan_outline,
        refined_steps_json=json.dumps(detail_steps_payload, ensure_ascii=False, indent=2),
    )
    hlg_system_msg = {"role": "system", "content": SYSTEM_PROMPT_ANALYST}
    hlg_detail_prompt_path = os.path.join(stage3_dir, "stage3_hlg_and_detail_independence_user_prompt.txt")
    hlg_detail_raw_path = os.path.join(stage3_dir, "stage3_hlg_and_detail_independence_raw_response.txt")
    write_text(hlg_detail_prompt_path, hlg_detail_prompt)
    stage1_frames_content = build_api_content(
        stage1_frames,
        api_cfg.embed_index_on_api_images,
        include_manifest=False,
        include_frame_labels=True,
    )

    last_hlg_detail_content = ""
    last_hlg_detail_errors: List[str] = []
    refined_high_level_goal: Optional[str] = None
    detail_by_id: Optional[Dict[int, str]] = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"[stage3] video_id={vid} hlg_and_detail_independence attempt={attempt}/{max_retries}")
        if attempt == 1:
            user_content = [{"type": "text", "text": hlg_detail_prompt}] + stage1_frames_content
        else:
            prefix = build_retry_prefix(last_hlg_detail_errors, last_hlg_detail_content)
            user_content = [{"type": "text", "text": prefix + hlg_detail_prompt}] + stage1_frames_content
        messages = [hlg_system_msg, {"role": "user", "content": user_content}]
        content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
        stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        stage_usage["total_tokens"] += usage.get("total_tokens", 0)
        stage_usage["api_calls"] += 1
        last_hlg_detail_content = content
        try:
            clean = extract_json_from_response(content)
            obj = json.loads(clean)
        except Exception as e:
            last_hlg_detail_errors = [f"JSON parse error: {e}"]
            continue
        if not isinstance(obj, dict):
            last_hlg_detail_errors = ["Combined HLG+detail_independence output is not an object."]
            continue


        candidate_hlg = obj.get("high_level_goal")
        normalized_hlg, hlg_errs = normalize_high_level_goal_text(candidate_hlg)
        if normalized_hlg is None:
            last_hlg_detail_errors = hlg_errs
            continue


        candidate_detail, detail_errs = _validate_stage3_detail_independence_result(obj, final_steps)
        if candidate_detail is None:
            last_hlg_detail_errors = detail_errs
            continue

        refined_high_level_goal = normalized_hlg
        detail_by_id = candidate_detail
        break

    write_text(hlg_detail_raw_path, last_hlg_detail_content)
    if refined_high_level_goal is None or detail_by_id is None:
        raise RuntimeError(
            f"Stage 3 failed to generate HLG+detail_independence for video_id={vid}: "
            + " | ".join(last_hlg_detail_errors[:10])
        )

    enriched_final_steps: List[Dict[str, Any]] = []
    for st in final_steps:
        if not isinstance(st, dict):
            continue
        sid = int(st.get("step_id", 0))
        independence = _normalize_independence_value(st.get("independence"))
        detail_independence: Optional[str] = None
        if sid > 1:
            if independence is None:
                raise RuntimeError(f"Missing independence before writing final Stage 3 plan for step_id={sid}.")
            detail_independence = detail_by_id.get(sid, "")
        enriched = _apply_dependency_fields(
            st,
            step_id=sid,
            independence=independence,
            detail_independence=detail_independence,
        )
        step_out_path = step_out_paths_by_id.get(sid)
        if step_out_path:
            write_json(step_out_path, enriched)
        enriched_final_steps.append(enriched)
    final_steps = enriched_final_steps

    final_plan = {"high_level_goal": refined_high_level_goal, "steps": final_steps}
    write_json(final_path, final_plan)

    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "stage3": {
                "status": "completed",
                "token_usage": stage_usage,
                "final_plan_path": os.path.relpath(final_path, video_out),
                "frame_index_note": (
                    "In this four-stage pipeline, critical_frames[*].frame_index is 1-based on EACH STEP CLIP's "
                    f"{int(sampling_cfg.max_frames)}-frame pool; see each step folder's frame_manifest.json."
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
        f"[stage3] video_id={vid} completed: steps={len(final_steps)} elapsed={format_duration(time.perf_counter() - t_start)}"
    )
    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: refine per-step using clips and generate keyframes.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root for generated stage outputs.")

    add_api_cli_args(parser, include_no_embed_index=True)

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
        run_stage3_for_video(
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
