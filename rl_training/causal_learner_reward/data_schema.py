

from __future__ import annotations

import json
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_ID = "qa_reward_schema"
RUBRIC_ID = "strict_multimodal_rubric"
DATA_SOURCE = "causal_learner"
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_MIN_FRAMES = 4
DEFAULT_VIDEO_MAX_FRAMES = 256

PathRewriteRule = Tuple[str, str, str]


@dataclass(frozen=True)
class TaskFieldSpec:
    task_id: str
    task_name_prefix: str
    required_keys: Tuple[str, ...]
    optional_keys: Tuple[str, ...] = ()


TASK_FIELD_SPECS: Dict[str, TaskFieldSpec] = {
    "Task_01": TaskFieldSpec(
        task_id="Task_01",
        task_name_prefix="Task_01_Spatial_Precondition",
        required_keys=("spatial_preconditions", "spatial_preconditions_pts", "step_goal", "patient"),
    ),
    "Task_02": TaskFieldSpec(
        task_id="Task_02",
        task_name_prefix="Task_02_Affordance_Precondition",
        required_keys=("affordance_preconditions", "affordance_preconditions_pts", "step_goal", "patient"),
    ),
    "Task_06": TaskFieldSpec(
        task_id="Task_06",
        task_name_prefix="Task_06_Spatial_Postcondition",
        required_keys=("spatial_postconditions", "spatial_postconditions_pts", "step_goal", "patient"),
    ),
    "Task_07": TaskFieldSpec(
        task_id="Task_07",
        task_name_prefix="Task_07_Affordance_Postcondition",
        required_keys=("affordance_postconditions", "affordance_postconditions_pts", "step_goal", "patient"),
    ),
    "Task_18": TaskFieldSpec(
        task_id="Task_18",
        task_name_prefix="Task_18_Bad_Plan_Diagnosis_And_Repair",
        required_keys=(
            "high_level_goal",
            "bad_plan_steps",
            "repair_steps",
            "flaw_step",
            "flaw_type",
            "prefix_end_step_id",
        ),
    ),
    "Task_19": TaskFieldSpec(
        task_id="Task_19",
        task_name_prefix="Task_19_Counterfactual_Outcome",
        required_keys=(
            "step_goal",
            "counterfactual_challenge_question",
            "expected_challenge_outcome",
            "expected_outcome",
        ),
    ),
    "Task_20": TaskFieldSpec(
        task_id="Task_20",
        task_name_prefix="Task_20_Failure_Recovery",
        required_keys=("step_goal", "failure_reason", "recovery_strategy"),
        optional_keys=("counterfactual_context", "counterfactual_outcome"),
    ),
}


def task_id_from_name(task_name: str) -> str:
    match = re.search(r"Task_(\d{2})", task_name or "")
    if not match:
        return ""
    return f"Task_{match.group(1)}"


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def clean_question_text(value: Any) -> str:
    text = ensure_text(value)
    text = re.sub(r"(?i)<\s*/?\s*video\s*>", "", text)
    return text.strip()


def strip_file_uri(path: str) -> str:
    return ensure_text(path).removeprefix("file://")


def to_file_uri(path: str) -> str:
    normalized = strip_file_uri(path)
    return f"file://{normalized}"


def stable_uid(*parts: Any) -> str:
    payload = "\n".join(ensure_text(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def single_video_path(value: Any) -> str:
    if isinstance(value, str):
        path = ensure_text(value)
        if not path:
            raise ValueError("video path is empty")
        return path
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected exactly one video, got {len(value)}")
        if not isinstance(value[0], str):
            raise ValueError(f"single video item must be a string, got {type(value[0]).__name__}")
        path = ensure_text(value[0])
        if not path:
            raise ValueError("video path is empty")
        return path
    raise ValueError(f"expected video to be a string or single-item list, got {type(value).__name__}")


def _require_keys(task_id: str, llm_fields: Mapping[str, Any]) -> None:
    spec = TASK_FIELD_SPECS[task_id]
    missing = []
    for key in spec.required_keys:
        value = llm_fields.get(key)
        if value is None or value == "" or value == []:
            missing.append(key)
    if missing:
        raise ValueError(f"{task_id} missing required llm_fields: {', '.join(missing)}")


def normalize_reward_fields(task_id: str, llm_fields: Mapping[str, Any]) -> Dict[str, Any]:


    if task_id not in TASK_FIELD_SPECS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    _require_keys(task_id, llm_fields)

    if task_id == "Task_01":
        return {
            "target_text": ensure_text(llm_fields["spatial_preconditions"]),
            "target_facts": [ensure_text(x) for x in ensure_list(llm_fields["spatial_preconditions_pts"])],
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "patient": ensure_text(llm_fields["patient"]),
            "fact_domain": "spatial",
            "fact_stage": "precondition",
        }
    if task_id == "Task_02":
        return {
            "target_text": ensure_text(llm_fields["affordance_preconditions"]),
            "target_facts": [ensure_text(x) for x in ensure_list(llm_fields["affordance_preconditions_pts"])],
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "patient": ensure_text(llm_fields["patient"]),
            "fact_domain": "affordance",
            "fact_stage": "precondition",
        }
    if task_id == "Task_06":
        return {
            "target_text": ensure_text(llm_fields["spatial_postconditions"]),
            "target_facts": [ensure_text(x) for x in ensure_list(llm_fields["spatial_postconditions_pts"])],
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "patient": ensure_text(llm_fields["patient"]),
            "fact_domain": "spatial",
            "fact_stage": "postcondition",
        }
    if task_id == "Task_07":
        return {
            "target_text": ensure_text(llm_fields["affordance_postconditions"]),
            "target_facts": [ensure_text(x) for x in ensure_list(llm_fields["affordance_postconditions_pts"])],
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "patient": ensure_text(llm_fields["patient"]),
            "fact_domain": "affordance",
            "fact_stage": "postcondition",
        }
    if task_id == "Task_18":
        return {
            "high_level_goal": ensure_text(llm_fields["high_level_goal"]),
            "bad_plan_steps": [ensure_text(x) for x in ensure_list(llm_fields["bad_plan_steps"])],
            "repair_steps": [ensure_text(x) for x in ensure_list(llm_fields["repair_steps"])],
            "flaw_step": int(llm_fields["flaw_step"]),
            "flaw_type": ensure_text(llm_fields["flaw_type"]),
            "prefix_end_step_id": int(llm_fields["prefix_end_step_id"]),
        }
    if task_id == "Task_19":
        return {
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "counterfactual_condition": ensure_text(llm_fields["counterfactual_challenge_question"]),
            "expected_challenge_outcome": ensure_text(llm_fields["expected_challenge_outcome"]),
            "expected_outcome": ensure_text(llm_fields["expected_outcome"]),
        }
    if task_id == "Task_20":
        fields = {
            "step_goal": ensure_text(llm_fields["step_goal"]),
            "failure_reason": ensure_text(llm_fields["failure_reason"]),
            "recovery_strategy": ensure_text(llm_fields["recovery_strategy"]),
        }
        for key in TASK_FIELD_SPECS[task_id].optional_keys:
            value = ensure_text(llm_fields.get(key))
            if value:
                fields[key] = value
        return fields

    raise ValueError(f"Unsupported task_id: {task_id}")


def parse_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        data = json.loads(ground_truth)
        if not isinstance(data, dict):
            raise ValueError("ground_truth JSON must decode to a dict")
        return data
    raise ValueError(f"ground_truth must be dict or JSON string, got {type(ground_truth).__name__}")


def build_ground_truth(row: Mapping[str, Any], task_id: Optional[str] = None) -> str:
    meta = row.get("meta") or {}
    task_name = ensure_text(meta.get("task_name"))
    resolved_task_id = task_id or task_id_from_name(task_name)
    if resolved_task_id not in TASK_FIELD_SPECS:
        raise ValueError(f"Unsupported or missing task_id: {resolved_task_id}")

    conversations = row.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError("conversations must contain human question and GPT answer")

    answer = ensure_text(conversations[1].get("value"))
    if not answer:
        raise ValueError("reference answer is empty")

    reward_fields = normalize_reward_fields(resolved_task_id, meta.get("llm_fields") or {})
    payload = {
        "schema_id": SCHEMA_ID,
        "task_id": resolved_task_id,
        "task_name": task_name,
        "answer": answer,
        "reward_fields": reward_fields,
    }
    return json.dumps(payload, ensure_ascii=False)


def apply_video_prefix_map(path: str, prefix_map: Sequence[Tuple[str, ...]] = ()) -> str:
    resolved = strip_file_uri(path)
    prefix_rules: List[PathRewriteRule] = []
    substr_rules: List[PathRewriteRule] = []
    for rule in prefix_map:
        if len(rule) == 2:
            kind, src, dst = "prefix", rule[0], rule[1]
        elif len(rule) == 3:
            kind, src, dst = rule
        else:
            raise ValueError(f"Invalid path rewrite rule: {rule!r}")
        if kind == "substr":
            substr_rules.append((kind, src, dst))
        elif kind == "prefix":
            prefix_rules.append((kind, src, dst))
        else:
            raise ValueError(f"Invalid path rewrite kind: {kind!r}")

    for _kind, src, dst in sorted(prefix_rules, key=lambda item: len(item[1]), reverse=True):
        src = src.rstrip("/")
        dst = dst.rstrip("/")
        if resolved == src or resolved.startswith(src + "/"):
            resolved = dst + resolved[len(src):]
            break
    for _kind, src, dst in substr_rules:
        if src:
            resolved = resolved.replace(src, dst)
    return resolved


def build_parquet_row(
    row: Mapping[str, Any],
    *,
    index: int,
    source_package: str,
    prefix_map: Sequence[Tuple[str, ...]] = (),
    task_id: Optional[str] = None,
    prompt_prefix: str = "<video>\n",
    source_jsonl: Optional[str] = None,
    source_line: Optional[int] = None,
    split: Optional[str] = None,
    fps: float = DEFAULT_VIDEO_FPS,
    min_frames: int = DEFAULT_VIDEO_MIN_FRAMES,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
) -> Dict[str, Any]:


    original_id = ensure_text(row.get("id"))
    if not original_id:
        raise ValueError("row id is missing")

    meta = row.get("meta") or {}
    task_name = ensure_text(meta.get("task_name"))
    resolved_task_id = task_id or task_id_from_name(task_name)
    if resolved_task_id not in TASK_FIELD_SPECS:
        raise ValueError(f"Unsupported or missing task_id: {resolved_task_id}")

    conversations = row.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError("conversations must contain human question and GPT answer")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise ValueError("conversations must be a human/GPT two-turn pair")

    image_items = ensure_list(row.get("image"))
    if image_items:
        raise ValueError("image field is non-empty; current causal RL parquet expects video-only samples")

    question = clean_question_text(conversations[0].get("value"))
    answer = ensure_text(conversations[1].get("value"))
    if not question or not answer:
        raise ValueError("question or answer is empty")
    if re.search(r"(?i)<\s*/?\s*image\s*>", question):
        raise ValueError("question contains an image placeholder but row has no supported image input")

    raw_video_path = single_video_path(row.get("video"))
    video_path = apply_video_prefix_map(raw_video_path, prefix_map)
    video_uri = to_file_uri(video_path)
    uid = stable_uid(source_package, resolved_task_id, source_jsonl or "", source_line or "", original_id)
    prompt_content = prompt_prefix + question
    if prompt_content.count("<video>") != 1:
        raise ValueError("converted prompt must contain exactly one <video> placeholder")

    return {
        "uid": uid,
        "prompt": [{"role": "user", "content": prompt_content}],
        "data_source": DATA_SOURCE,
        "reward_model": {"style": "rule", "ground_truth": build_ground_truth(row, resolved_task_id)},
        "videos": [{"video": video_uri, "fps": fps, "min_frames": min_frames, "max_frames": max_frames}],
        "extra_info": {
            "index": index,
            "uid": uid,
            "original_id": original_id,
            "task_id": resolved_task_id,
            "task_name": task_name,
            "source_package": source_package,
            "source_jsonl": ensure_text(source_jsonl),
            "source_line": source_line,
            "source_path": ensure_text(meta.get("source_path")),
            "item_dir": ensure_text(meta.get("item_dir")),
            "evidence_type": ensure_text(meta.get("evidence_type")),
            "split": ensure_text(split),
            "question": question,
            "reference_answer": answer,
            "raw_video_path": raw_video_path,
            "video_path": video_path,
            "resolved_video_path": video_path,
            "video_uri": video_uri,
            "reward_schema_id": SCHEMA_ID,
            "rubric_id": RUBRIC_ID,
        },
    }


def validate_parquet_row(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], dict):
        errors.append("prompt must be a non-empty list of message dicts")
    else:
        content = prompt[0].get("content")
        if not isinstance(content, str):
            errors.append("prompt[0].content must be a string")
        elif content.count("<video>") != 1:
            errors.append("prompt[0].content must contain exactly one <video> placeholder")

    if row.get("data_source") != DATA_SOURCE:
        errors.append(f"data_source must be {DATA_SOURCE!r}")

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        errors.append("reward_model must be a dict")
    else:
        try:
            ground_truth = parse_ground_truth(reward_model.get("ground_truth"))
            task_id = ground_truth.get("task_id")
            if task_id not in TASK_FIELD_SPECS:
                errors.append(f"unsupported ground_truth task_id: {task_id}")
            if ground_truth.get("schema_id") != SCHEMA_ID:
                errors.append("ground_truth schema_id mismatch")
            if not isinstance(ground_truth.get("reward_fields"), dict):
                errors.append("ground_truth.reward_fields must be a dict")
        except Exception as exc:
            errors.append(f"invalid reward_model.ground_truth: {exc}")

    videos = row.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        errors.append("videos must be a single-item list")
    elif not all(isinstance(item, dict) and item.get("video") for item in videos):
        errors.append("the videos item must be a dict with a non-empty 'video' field")

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        errors.append("extra_info must be a dict")
    else:
        for key in (
            "task_id",
            "uid",
            "question",
            "reference_answer",
            "raw_video_path",
            "resolved_video_path",
            "video_uri",
        ):
            if not extra_info.get(key):
                errors.append(f"extra_info.{key} is missing")

    return errors


def parse_prefix_map(items: Optional[Iterable[str]]) -> List[PathRewriteRule]:
    pairs: List[PathRewriteRule] = []
    for item in items or []:
        kind = "prefix"
        if item.startswith("SUBSTR:"):
            kind = "substr"
            item = item[len("SUBSTR:") :]
        if "=" not in item:
            raise ValueError(f"Invalid prefix map {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        pairs.append((kind, old, new))
    return pairs


def source_package_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return ""
