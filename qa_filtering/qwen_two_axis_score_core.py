from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable

from qa_filter_io import QASample


QWEN_TWO_AXIS_SCORE_SYSTEM_PROMPT = (
    "You are Qwen3.5-397B-A17B acting as a strict multimodal QA filtering judge. "
    "Evaluate each generated QA instance in one pass for accurate visual grounding "
    "and general logical coherence. Use only the supplied visual evidence and source "
    "context. Return valid JSON only."
)


QWEN_TWO_AXIS_SCORE_RUBRIC = """
Score two axes from 0 to 100:

A high-quality item must satisfy both axes. Fluent wording cannot compensate for
weak visual grounding, and visually grounded fragments cannot compensate for a
question-answer mismatch or incoherent causal/temporal relation.

visual_grounding_score:
- 90-100: all visible objects, actions, states, and ordering are directly supported.
- 70-89: mostly grounded, with minor omissions that do not change the answer.
- 40-69: partially grounded but ambiguous, underspecified, or weakly supported.
- 1-39: major unsupported visual claim, wrong object/action/state, or wrong evidence.
- 0: missing/unreadable evidence or answer is wholly unrelated to evidence.

logical_coherence_score:
- 90-100: question, answer, task, source context, and causal/temporal relation are coherent.
- 70-89: coherent overall, with minor wording or specificity issues.
- 40-69: partially coherent but missing important causal/temporal support.
- 1-39: contradiction, wrong step, impossible mechanism, or question-answer mismatch.
- 0: invalid, nonsensical, or unrelated QA.

Use hard failure tags whenever applicable:
visual_hallucination, missing_visual_evidence, question_answer_mismatch,
unsupported_causal_claim, contradiction, wrong_step, impossible_physics,
timeline_mismatch.

Return exactly:
{
  "visual_grounding_score": 0-100,
  "logical_coherence_score": 0-100,
  "visual_grounding_rationale": "short evidence-based reason",
  "logical_coherence_rationale": "short evidence-based reason",
  "score_confidence": "low" or "medium" or "high",
  "major_failure_tags": ["none"] or failure tags
}
"""


QWEN_SCORE_FORMULA = (
    "overall_score = min(mean(visual_grounding_score, logical_coherence_score), "
    "min_axis + 20, hard_failure_caps, missing_rationale_cap)"
)


QWEN_HARD_FAILURE_CAPS = {
    "missing_visual_evidence": 0.0,
    "visual_hallucination": 24.0,
    "question_answer_mismatch": 34.0,
    "unsupported_causal_claim": 44.0,
    "contradiction": 34.0,
    "wrong_step": 34.0,
    "impossible_physics": 24.0,
    "timeline_mismatch": 34.0,
    "parse_error": 0.0,
    "preflight_error": 0.0,
}


def build_qwen_two_axis_score_messages(sample: QASample, base64_images: list[str]) -> list[dict[str, Any]]:
    source_context = json.dumps(sample.source_context or sample.llm_fields, ensure_ascii=False, indent=2)
    user_text = "\n\n".join(
        [
            QWEN_TWO_AXIS_SCORE_RUBRIC.strip(),
            f"Task name:\n{sample.task_name}",
            f"Evidence type:\n{sample.evidence_type}",
            f"Sample ID:\n{sample.sample_id}",
            f"Question:\n{sample.question}",
            f"Generated answer:\n{sample.answer}",
            f"Source context:\n{source_context}",
            "Return valid JSON only.",
        ]
    )
    return [
        {"role": "system", "content": QWEN_TWO_AXIS_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Ordered visual evidence follows."},
                *image_blocks(base64_images),
                {"type": "text", "text": user_text},
            ],
        },
    ]


def image_blocks(base64_images: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for encoded in base64_images:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                },
            }
        )
    return blocks


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return parsed


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a score")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("non-finite score")
    return max(0.0, min(100.0, round(score, 3)))


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    tags = [str(tag).strip().lower() for tag in raw if str(tag).strip()]
    tags = [tag for tag in tags if tag not in {"none", "n/a", "na", "null"}]
    return tags or ["none"]


def parse_qwen_two_axis_score_response(raw_text: str, *, task_name: str = "") -> dict[str, Any]:
    warnings: list[str] = []
    try:
        parsed = _extract_json_object(raw_text)
        visual_score = _score(parsed.get("visual_grounding_score"))
        logical_score = _score(parsed.get("logical_coherence_score"))
        visual_rationale = str(parsed.get("visual_grounding_rationale") or "").strip()
        logical_rationale = str(parsed.get("logical_coherence_rationale") or "").strip()
        tags = _tags(parsed.get("major_failure_tags") or parsed.get("failure_tags"))
        confidence = str(parsed.get("score_confidence") or "medium").strip().lower()
    except Exception as exc:
        return {
            "visual_grounding_score": 0.0,
            "logical_coherence_score": 0.0,
            "overall_score": 0.0,
            "visual_grounding_rationale": "",
            "logical_coherence_rationale": "",
            "score_confidence": "low",
            "major_failure_tags": ["parse_error"],
            "score_formula": QWEN_SCORE_FORMULA,
            "score_warnings": [f"parse_error:{type(exc).__name__}:{exc}"],
            "parse_error": True,
            "task_name": task_name,
        }

    mean_score = round((visual_score + logical_score) / 2.0, 3)
    min_axis = min(visual_score, logical_score)
    caps = [mean_score, min_axis + 20.0]
    active_tags = [tag for tag in tags if tag != "none"]
    for tag in active_tags:
        if tag in QWEN_HARD_FAILURE_CAPS:
            caps.append(QWEN_HARD_FAILURE_CAPS[tag])
            warnings.append(f"hard_cap:{tag}<={QWEN_HARD_FAILURE_CAPS[tag]}")
    if not visual_rationale or not logical_rationale:
        caps.append(60.0)
        warnings.append("missing_rationale_cap<=60")
    overall = round(max(0.0, min(100.0, min(caps))), 3)

    return {
        "visual_grounding_score": visual_score,
        "logical_coherence_score": logical_score,
        "overall_score": overall,
        "visual_grounding_rationale": visual_rationale,
        "logical_coherence_rationale": logical_rationale,
        "score_confidence": confidence if confidence in {"low", "medium", "high"} else "medium",
        "major_failure_tags": tags,
        "score_formula": QWEN_SCORE_FORMULA,
        "score_warnings": warnings,
        "parse_error": False,
        "task_name": task_name,
    }


def zero_score(reason: str) -> dict[str, Any]:
    return {
        "visual_grounding_score": 0.0,
        "logical_coherence_score": 0.0,
        "overall_score": 0.0,
        "visual_grounding_rationale": reason,
        "logical_coherence_rationale": reason,
        "score_confidence": "low",
        "major_failure_tags": ["preflight_error"],
        "score_formula": QWEN_SCORE_FORMULA,
        "score_warnings": [reason],
        "parse_error": False,
        "task_name": "",
    }


def _is_exception_instance(exc: BaseException, exc_type: Any) -> bool:
    return isinstance(exc_type, type) and issubclass(exc_type, BaseException) and isinstance(exc, exc_type)


def _runtime_exception_types(cv2_error: Any) -> tuple[type[BaseException], ...]:
    out: tuple[type[BaseException], ...] = (RuntimeError,)
    if isinstance(cv2_error, type) and issubclass(cv2_error, BaseException):
        out = out + (cv2_error,)
    return out


def run_qwen_two_axis_score_qa(
    llm: Any,
    sample: QASample,
    *,
    resolve_evidence_frames: Callable[[Any], list[str]],
    black_frame_error: Any,
    cv2_error: Any,
    logger: Any = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    try:
        base64_images = resolve_evidence_frames(sample)
    except Exception as exc:
        if _is_exception_instance(exc, black_frame_error):
            return _with_evidence_metadata(zero_score(f"black-frame evidence: {exc}"), 0, "")
        if isinstance(exc, (FileNotFoundError, ValueError)):
            return _with_evidence_metadata(zero_score(f"evidence error: {exc}"), 0, "")
        if isinstance(exc, _runtime_exception_types(cv2_error)):
            return _with_evidence_metadata(zero_score(f"evidence runtime error: {exc}"), 0, "")
        return _with_evidence_metadata(zero_score(f"unexpected evidence error: {exc}"), 0, "")
    if not base64_images:
        return _with_evidence_metadata(zero_score("no visual evidence extracted"), 0, "")
    messages = build_qwen_two_axis_score_messages(sample, base64_images)
    raw = llm.call_multimodal(messages=messages, max_tokens=max_tokens, reasoning_effort="high")
    score = parse_qwen_two_axis_score_response(raw, task_name=sample.task_name)
    return _with_evidence_metadata(score, len(base64_images), hashlib.sha256(raw.encode("utf-8")).hexdigest())


def _with_evidence_metadata(score: dict[str, Any], frame_count: int, judge_hash: str) -> dict[str, Any]:
    out = dict(score)
    out["evidence_frame_count"] = frame_count
    out["judge_response_sha256"] = judge_hash
    return out
