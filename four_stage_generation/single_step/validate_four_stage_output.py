from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def _read_json(path: str, errors: List[str]) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        errors.append(f"Cannot read JSON: {path}: {type(exc).__name__}: {exc}")
        return None


def _nonempty_steps(obj: Any, label: str, errors: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        errors.append(f"{label} must be a JSON object.")
        return []
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(f"{label}.steps must be a non-empty list.")
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"{label}.steps[{idx}] must be an object.")
            continue
        sid = step.get("step_id")
        if not isinstance(sid, int):
            errors.append(f"{label}.steps[{idx}].step_id must be an integer.")
        elif sid in seen:
            errors.append(f"{label}.steps contains duplicate step_id={sid}.")
        else:
            seen.add(sid)
        out.append(step)
    return out


def validate_four_stage_video_output_dir(video_out: str, check_deps: bool = False) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    root = os.path.abspath(video_out)
    required = {
        "stage1": os.path.join(root, "stage1", "draft_plan.json"),
        "stage2": os.path.join(root, "stage2", "step_segments.json"),
        "stage3": os.path.join(root, "stage3", "causal_plan_with_keyframes.json"),
        "stage4": os.path.join(root, "stage4", "atomic_plan_with_clips.json"),
    }
    for label, path in required.items():
        if not os.path.exists(path):
            errors.append(f"Missing {label} output: {path}")
    if errors:
        return False, errors, warnings

    stage1 = _read_json(required["stage1"], errors)
    stage2 = _read_json(required["stage2"], errors)
    stage3 = _read_json(required["stage3"], errors)
    stage4 = _read_json(required["stage4"], errors)
    s1_steps = _nonempty_steps(stage1, "stage1", errors)
    s2_steps = _nonempty_steps(stage2, "stage2", errors)
    s3_steps = _nonempty_steps(stage3, "stage3", errors)
    s4_steps = _nonempty_steps(stage4, "stage4", errors)
    counts = {len(s1_steps), len(s2_steps), len(s3_steps), len(s4_steps)}
    if len(counts) > 1:
        errors.append(
            "Stage step counts disagree: "
            f"stage1={len(s1_steps)}, stage2={len(s2_steps)}, stage3={len(s3_steps)}, stage4={len(s4_steps)}"
        )
    for idx, step in enumerate(s4_steps):
        actions = step.get("atomic_actions")
        if not isinstance(actions, list) or not actions:
            errors.append(f"stage4.steps[{idx}].atomic_actions must be a non-empty list.")
    return not errors, errors, warnings
