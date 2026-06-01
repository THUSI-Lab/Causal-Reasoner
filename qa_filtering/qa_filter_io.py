from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TASK_DIR_RE = re.compile(r"^Task[_-]?\d+", re.IGNORECASE)


@dataclass(frozen=True)
class QASample:
    task_name: str
    row: dict[str, Any]
    row_index: int
    source_file: Path
    sample_id: str
    question: str
    answer: str
    evidence_type: str
    image: list[str]
    video: str
    llm_fields: dict[str, Any]
    source_context: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected object row, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def iter_task_data_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    candidates = sorted(input_dir.glob("Task_*/data.jsonl")) + sorted(input_dir.glob("Task-*/data.jsonl"))
    if candidates:
        return candidates
    direct = input_dir / "data.jsonl"
    if direct.exists():
        return [direct]
    return sorted(path for path in input_dir.rglob("data.jsonl") if TASK_DIR_RE.match(path.parent.name))


def infer_task_name(path: Path, row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for value in (
        row.get("task_name"),
        row.get("task"),
        meta.get("task_name"),
        meta.get("task"),
        path.parent.name,
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "Task_unknown"


def load_plan(path: str | os.PathLike[str] | None) -> Any:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_prefix_step_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "")
    m = re.search(r"(?:step|prefix|atomic[_-]?step)[^\d]*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def match_step(plan: Any, step_index: int | None) -> dict[str, Any]:
    if plan is None or step_index is None:
        return {}
    candidates: list[Any] = []
    if isinstance(plan, dict):
        for key in (
            "steps",
            "atomic_steps",
            "atomic_actions",
            "plan",
            "refined_plan",
            "causal_plan",
            "stage3_steps",
            "stage4_atomic_actions",
        ):
            value = plan.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if not candidates:
            for value in plan.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    candidates.extend(value)
    elif isinstance(plan, list):
        candidates = plan

    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_indices = [
            item.get("step_index"),
            item.get("step_id"),
            item.get("index"),
            item.get("id"),
            item.get("atomic_step_index"),
        ]
        if any(extract_prefix_step_index(idx) == step_index for idx in item_indices):
            return item
    if 0 <= step_index < len(candidates) and isinstance(candidates[step_index], dict):
        return candidates[step_index]
    if 1 <= step_index <= len(candidates) and isinstance(candidates[step_index - 1], dict):
        return candidates[step_index - 1]
    return {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _conversation_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    seq = row.get("conversations") or row.get("messages") or row.get("conversation")
    if not isinstance(seq, list):
        return "", ""
    question = ""
    answer = ""
    for item in seq:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from") or item.get("role") or "").lower()
        text = str(item.get("value") or item.get("content") or item.get("text") or "").strip()
        if not text:
            continue
        if role in {"human", "user", "question"} and not question:
            question = text
        elif role in {"gpt", "assistant", "model", "answer"}:
            answer = text
    return question, answer


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
            elif isinstance(item, dict):
                for key in ("path", "image", "video", "url"):
                    text = str(item.get(key) or "").strip()
                    if text:
                        out.append(text)
                        break
        return out
    return []


def _resolve_media_paths(paths: list[str], source_file: Path) -> list[str]:
    resolved: list[str] = []
    base_dirs = [source_file.parent, source_file.parent.parent]
    for raw in paths:
        text = str(raw).strip()
        if not text:
            continue
        p = Path(text).expanduser()
        if p.is_absolute() or text.startswith("data:") or text.startswith("http://") or text.startswith("https://"):
            resolved.append(text)
            continue
        found = None
        for base in base_dirs:
            candidate = (base / p).resolve()
            if candidate.exists():
                found = candidate
                break
        resolved.append(str(found) if found else text)
    return resolved


def _extract_media(row: dict[str, Any], source_file: Path) -> tuple[list[str], str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    image_paths = (
        _as_list(row.get("image"))
        + _as_list(row.get("images"))
        + _as_list(row.get("image_path"))
        + _as_list(meta.get("image"))
        + _as_list(meta.get("images"))
        + _as_list(meta.get("image_path"))
    )
    video_values = (
        _as_list(row.get("video"))
        + _as_list(row.get("videos"))
        + _as_list(row.get("video_path"))
        + _as_list(meta.get("video"))
        + _as_list(meta.get("videos"))
        + _as_list(meta.get("video_path"))
    )
    image = _resolve_media_paths(image_paths, source_file)
    videos = _resolve_media_paths(video_values, source_file)
    if len(videos) > 1:
        video = json.dumps(videos, ensure_ascii=False)
    else:
        video = videos[0] if videos else ""
    return image, video


def _infer_evidence_type(row: dict[str, Any], image: list[str], video: str) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    llm_fields = row.get("llm_fields") if isinstance(row.get("llm_fields"), dict) else {}
    explicit = _first_nonempty(
        row.get("evidence_type"),
        meta.get("evidence_type"),
        llm_fields.get("evidence_type"),
        meta.get("qa_evidence_type"),
    )
    if explicit:
        return explicit
    if video:
        try:
            parsed = json.loads(video)
            if isinstance(parsed, list) and len(parsed) == 2:
                return "video_clip_pair"
        except Exception:
            pass
        text = json.dumps(row, ensure_ascii=False).lower()
        if "prefix" in text or "before" in text:
            return "video_prefix"
        return "video_clip"
    if image:
        return "keyframe_single"
    return "unknown"


def build_llm_fields(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    out: dict[str, Any] = {}
    for source in (row.get("llm_fields"), meta.get("llm_fields"), meta, row):
        if not isinstance(source, dict):
            continue
        for key in (
            "task_name",
            "question_type",
            "source_context",
            "plan_step",
            "step",
            "step_index",
            "action",
            "object",
            "precondition",
            "effect",
            "causal_relation",
            "counterfactual",
            "answer_rationale",
            "evidence_type",
        ):
            value = source.get(key)
            if value not in (None, "", [], {}):
                out.setdefault(key, value)

    plan_path = _first_nonempty(
        row.get("final_plan_path"),
        row.get("plan_path"),
        meta.get("final_plan_path"),
        meta.get("plan_path"),
        meta.get("source_plan_path"),
    )
    plan = load_plan(plan_path)
    step_index = extract_prefix_step_index(
        out.get("step_index") or out.get("step") or out.get("plan_step") or meta.get("step_id")
    )
    matched_step = match_step(plan, step_index)
    if matched_step:
        out.setdefault("matched_plan_step", matched_step)
    if plan_path:
        out.setdefault("plan_path_available", bool(plan))
    return out


def source_context_issue(llm_fields: dict[str, Any]) -> str:
    if not isinstance(llm_fields, dict) or not llm_fields:
        return "missing_llm_fields"
    has_context = any(
        key in llm_fields and llm_fields[key] not in (None, "", [], {})
        for key in (
            "source_context",
            "matched_plan_step",
            "plan_step",
            "step",
            "action",
            "precondition",
            "effect",
            "causal_relation",
        )
    )
    return "" if has_context else "missing_source_context"


def row_to_sample(row: dict[str, Any], source_file: Path, row_index: int) -> QASample:
    task_name = infer_task_name(source_file, row)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    conversation_q, conversation_a = _conversation_question_answer(row)
    question = _first_nonempty(
        row.get("question"),
        row.get("prompt"),
        row.get("query"),
        meta.get("question"),
        conversation_q,
    )
    answer = _first_nonempty(
        row.get("answer"),
        row.get("response"),
        row.get("output"),
        row.get("target"),
        meta.get("answer"),
        meta.get("reference_answer"),
        conversation_a,
    )
    image, video = _extract_media(row, source_file)
    evidence_type = _infer_evidence_type(row, image, video)
    llm_fields = build_llm_fields(row)
    sample_id = _first_nonempty(
        row.get("id"),
        row.get("sample_id"),
        meta.get("id"),
        meta.get("sample_id"),
        f"{task_name}:{source_file.name}:{row_index}",
    )
    source_context = {
        key: value
        for key, value in llm_fields.items()
        if key
        in {
            "source_context",
            "matched_plan_step",
            "plan_step",
            "step",
            "action",
            "precondition",
            "effect",
            "causal_relation",
            "counterfactual",
            "answer_rationale",
        }
    }
    return QASample(
        task_name=task_name,
        row=row,
        row_index=row_index,
        source_file=source_file,
        sample_id=sample_id,
        question=question,
        answer=answer,
        evidence_type=evidence_type,
        image=image,
        video=video,
        llm_fields=llm_fields,
        source_context=source_context,
    )


def load_samples(input_dir: Path) -> list[QASample]:
    samples: list[QASample] = []
    for data_file in iter_task_data_files(input_dir):
        for row_index, row in enumerate(read_jsonl(data_file)):
            samples.append(row_to_sample(row, data_file, row_index))
    return samples


def preflight_sample(sample: QASample, *, require_source_context: bool = True) -> list[str]:
    issues: list[str] = []
    if not sample.question:
        issues.append("missing_question")
    if not sample.answer:
        issues.append("missing_answer")
    if sample.evidence_type == "unknown":
        issues.append("missing_evidence_type")
    if sample.evidence_type == "keyframe_single" and not sample.image:
        issues.append("missing_image_evidence")
    if sample.evidence_type in {"video_clip", "video_prefix", "video_clip_pair"} and not sample.video:
        issues.append("missing_video_evidence")
    if require_source_context:
        issue = source_context_issue(sample.llm_fields)
        if issue:
            issues.append(issue)
    return issues


def attach_filter_metadata(row: dict[str, Any], key: str, value: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta[key] = value
    out["meta"] = meta
    return out
