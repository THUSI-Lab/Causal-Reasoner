from __future__ import annotations

import json
from typing import Any

try:
    from .qa_filter_io import QASample
    from .qwen_two_axis_score_core import _extract_json_object, image_blocks
    from .unified_filter_core import _norm_status
except ImportError:
    from qa_filter_io import QASample
    from qwen_two_axis_score_core import _extract_json_object, image_blocks
    from unified_filter_core import _norm_status


PHYSICAL_LOGIC_AUDIT_SYSTEM_PROMPT = (
    "You are a strict autonomous physical-logic auditor for multimodal "
    "causal-planning QA. Reject violated preconditions, unsupported causal "
    "dependencies, impossible state transitions, hallucinated effects, timeline "
    "mismatches, and physical infeasibility. Return valid JSON only."
)


PHYSICAL_LOGIC_AUDIT_RUBRIC = """
Audit all dimensions:
1. Preconditions: required objects, states, contacts, poses, and prior actions exist.
2. Causal dependency: the claimed cause/effect is supported and not merely plausible.
3. State transition: before/after states are physically reachable and not invented.
4. Timeline consistency: no reversed order, skipped prerequisite, or wrong clip/step.
5. Hallucinated effects: no invented outcome, movement, damage, completion, or recovery.
6. Physical feasibility: the action and consequence are possible in the scene.
7. QA alignment: the answer directly answers the question.

Reject if any dimension fails. Return exactly:
{
  "decision": "ACCEPT" or "REJECT",
  "precondition_validity": "pass" or "fail",
  "causal_dependency": "pass" or "fail",
  "state_transition": "pass" or "fail",
  "timeline_consistency": "pass" or "fail",
  "hallucinated_effects": [],
  "physical_feasibility": "pass" or "fail",
  "qa_alignment": "pass" or "fail",
  "failure_tags": ["none"] or short snake_case failure tags,
  "rationale": "one concise evidence-based reason"
}
"""


PHYSICAL_HARD_REJECT_TAGS = {
    "violated_precondition",
    "unsupported_causal_dependency",
    "impossible_state_transition",
    "timeline_mismatch",
    "hallucinated_effect",
    "physical_infeasibility",
    "qa_misalignment",
    "missing_visual_evidence",
    "wrong_step",
    "parse_error",
    "preflight_error",
}


def build_physical_logic_audit_messages(sample: QASample, base64_images: list[str]) -> list[dict[str, Any]]:
    source_context = json.dumps(sample.source_context or sample.llm_fields, ensure_ascii=False, indent=2)
    user_text = "\n\n".join(
        [
            PHYSICAL_LOGIC_AUDIT_RUBRIC.strip(),
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
        {"role": "system", "content": PHYSICAL_LOGIC_AUDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Ordered visual evidence follows."},
                *image_blocks(base64_images),
                {"type": "text", "text": user_text},
            ],
        },
    ]


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


def parse_physical_logic_audit_response(raw_text: str, *, task_name: str = "") -> dict[str, Any]:
    try:
        parsed = _extract_json_object(raw_text)
    except Exception as exc:
        return {
            "decision": "REJECT",
            "precondition_validity": "fail",
            "causal_dependency": "fail",
            "state_transition": "fail",
            "timeline_consistency": "fail",
            "hallucinated_effects": [],
            "physical_feasibility": "fail",
            "qa_alignment": "fail",
            "failure_tags": ["parse_error"],
            "rationale": f"Could not parse audit JSON: {type(exc).__name__}: {exc}",
            "parse_error": True,
            "task_name": task_name,
        }

    fields = {
        "precondition_validity": _norm_status(parsed.get("precondition_validity")),
        "causal_dependency": _norm_status(parsed.get("causal_dependency")),
        "state_transition": _norm_status(parsed.get("state_transition")),
        "timeline_consistency": _norm_status(parsed.get("timeline_consistency")),
        "physical_feasibility": _norm_status(parsed.get("physical_feasibility")),
        "qa_alignment": _norm_status(parsed.get("qa_alignment")),
    }
    hallucinated_raw = parsed.get("hallucinated_effects") or []
    if isinstance(hallucinated_raw, str):
        hallucinated_effects = [hallucinated_raw] if hallucinated_raw.strip() else []
    elif isinstance(hallucinated_raw, list):
        hallucinated_effects = [str(item).strip() for item in hallucinated_raw if str(item).strip()]
    else:
        hallucinated_effects = []
    tags = _norm_tags(parsed.get("failure_tags") or parsed.get("major_failure_tags"))
    if hallucinated_effects and "hallucinated_effect" not in tags:
        tags = [tag for tag in tags if tag != "none"] + ["hallucinated_effect"]
    active_tags = [tag for tag in tags if tag != "none"]
    explicit = str(parsed.get("decision") or "").strip().upper()
    all_pass = all(value == "pass" for value in fields.values())
    hard_reject = any(tag in PHYSICAL_HARD_REJECT_TAGS for tag in active_tags)
    accept = explicit == "ACCEPT" and all_pass and not hallucinated_effects and not hard_reject
    if explicit not in {"ACCEPT", "REJECT"}:
        accept = all_pass and not hallucinated_effects and not active_tags
    if not accept and tags == ["none"]:
        tags = ["physical_logic_rejected"]
    rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip() or "Auditor did not provide a rationale."
    return {
        "decision": "ACCEPT" if accept else "REJECT",
        **fields,
        "hallucinated_effects": hallucinated_effects,
        "failure_tags": tags,
        "rationale": rationale,
        "parse_error": False,
        "task_name": task_name,
    }
