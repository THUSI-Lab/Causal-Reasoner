from __future__ import annotations

import json
from typing import Any

try:
    from .qa_filter_io import QASample
    from .qwen_two_axis_score_core import _extract_json_object, image_blocks
except ImportError:
    from qa_filter_io import QASample
    from qwen_two_axis_score_core import _extract_json_object, image_blocks


UNIFIED_FILTER_SYSTEM_PROMPT = (
    "You are a strict multimodal QA quality judge. Use only the supplied "
    "question, answer, source context, and visual evidence. Return valid JSON only."
)


UNIFIED_FILTER_RUBRIC = """
Judge whether this generated QA item should be accepted.

Accept only when all are true:
1. The answer is visually grounded in the attached evidence.
2. The question and answer are logically coherent and task-aligned.
3. The causal or temporal relation is supported by the source context.

Reject hallucinated objects/actions/effects, wrong-step answers, timeline
mismatches, unsupported causal claims, impossible physics, unreadable evidence,
or question-answer mismatches.

Return exactly:
{
  "decision": "ACCEPT" or "REJECT",
  "visual_grounding": "pass" or "fail",
  "logical_coherence": "pass" or "fail",
  "qa_alignment": "pass" or "fail",
  "failure_tags": ["none"] or short snake_case failure tags,
  "rationale": "one concise evidence-based reason"
}
"""


HARD_REJECT_TAGS = {
    "visual_hallucination",
    "missing_visual_evidence",
    "question_answer_mismatch",
    "unsupported_causal_claim",
    "contradiction",
    "wrong_step",
    "impossible_physics",
    "timeline_mismatch",
    "parse_error",
    "preflight_error",
}


def build_unified_filter_messages(sample: QASample, base64_images: list[str]) -> list[dict[str, Any]]:
    source_context = json.dumps(sample.source_context or sample.llm_fields, ensure_ascii=False, indent=2)
    user_text = "\n\n".join(
        [
            UNIFIED_FILTER_RUBRIC.strip(),
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
        {"role": "system", "content": UNIFIED_FILTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Ordered visual evidence follows."},
                *image_blocks(base64_images),
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _norm_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"pass", "passed", "yes", "true", "ok", "acceptable", "accept"}:
        return "pass"
    return "fail"


def _norm_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    tags = [str(tag).strip().lower() for tag in raw if str(tag).strip()]
    tags = [tag for tag in tags if tag not in {"none", "n/a", "na", "null"}]
    return tags or ["none"]


def parse_unified_filter_response(raw_text: str, *, task_name: str = "") -> dict[str, Any]:
    try:
        parsed = _extract_json_object(raw_text)
    except Exception as exc:
        return {
            "decision": "REJECT",
            "visual_grounding": "fail",
            "logical_coherence": "fail",
            "qa_alignment": "fail",
            "failure_tags": ["parse_error"],
            "rationale": f"Could not parse judge JSON: {type(exc).__name__}: {exc}",
            "parse_error": True,
            "task_name": task_name,
        }

    tags = _norm_tags(parsed.get("failure_tags") or parsed.get("major_failure_tags"))
    active_tags = [tag for tag in tags if tag != "none"]
    visual = _norm_status(parsed.get("visual_grounding"))
    logical = _norm_status(parsed.get("logical_coherence"))
    alignment = _norm_status(parsed.get("qa_alignment"))
    explicit = str(parsed.get("decision") or "").strip().upper()
    hard_reject = any(tag in HARD_REJECT_TAGS for tag in active_tags)
    accept = explicit == "ACCEPT" and visual == "pass" and logical == "pass" and alignment == "pass" and not hard_reject
    if explicit not in {"ACCEPT", "REJECT"}:
        accept = visual == "pass" and logical == "pass" and alignment == "pass" and not active_tags
    if not accept and tags == ["none"]:
        tags = ["judge_rejected"]
    rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip() or "Judge did not provide a rationale."
    return {
        "decision": "ACCEPT" if accept else "REJECT",
        "visual_grounding": visual,
        "logical_coherence": logical,
        "qa_alignment": alignment,
        "failure_tags": tags,
        "rationale": rationale,
        "parse_error": False,
        "task_name": task_name,
    }
