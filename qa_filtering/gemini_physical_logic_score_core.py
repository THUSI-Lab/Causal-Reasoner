from __future__ import annotations

import json
import math
from typing import Any

try:
    from .qa_filter_io import QASample
    from .qwen_two_axis_score_core import _extract_json_object, image_blocks
except ImportError:
    from qa_filter_io import QASample
    from qwen_two_axis_score_core import _extract_json_object, image_blocks


GEMINI_PHYSICAL_LOGIC_SCORE_SYSTEM_PROMPT = (
    "You are Gemini acting as a strict physical-logic scorer for multimodal "
    "causal-planning QA. Audit preconditions, causal dependencies, state "
    "transitions, timelines, and physical feasibility. Return valid JSON only."
)


GEMINI_PHYSICAL_LOGIC_SCORE_RUBRIC = """
Score each axis from 0 to 100:
precondition_validity_score: required states and prerequisites exist before the claimed action/effect.
causal_dependency_score: the source-supported causal link is necessary or strongly supported.
state_transition_score: before/after states are physically reachable and not invented.
timeline_consistency_score: event order and clip/step assignment are consistent.
physical_feasibility_score: the action and consequence are possible in the scene.

Use major failure tags when applicable:
violated_precondition, unsupported_causal_dependency, impossible_state_transition,
timeline_mismatch, hallucinated_effect, physical_infeasibility, qa_misalignment,
missing_visual_evidence.

Return exactly:
{
  "precondition_validity_score": 0-100,
  "causal_dependency_score": 0-100,
  "state_transition_score": 0-100,
  "timeline_consistency_score": 0-100,
  "physical_feasibility_score": 0-100,
  "precondition_rationale": "short evidence-based reason",
  "causal_dependency_rationale": "short evidence-based reason",
  "state_transition_rationale": "short evidence-based reason",
  "timeline_rationale": "short evidence-based reason",
  "physical_feasibility_rationale": "short evidence-based reason",
  "hallucinated_effects": [],
  "score_confidence": "low" or "medium" or "high",
  "major_failure_tags": ["none"] or failure tags
}
"""


GEMINI_SCORE_FORMULA = (
    "overall_physical_logic_score = min(mean(five_axis_scores), min_axis + 15, "
    "low_axis_caps, hard_failure_caps, missing_rationale_cap)"
)


GEMINI_HARD_FAILURE_CAPS = {
    "missing_visual_evidence": 0.0,
    "violated_precondition": 34.0,
    "unsupported_causal_dependency": 44.0,
    "impossible_state_transition": 24.0,
    "timeline_mismatch": 34.0,
    "hallucinated_effect": 24.0,
    "physical_infeasibility": 24.0,
    "qa_misalignment": 34.0,
    "parse_error": 0.0,
    "preflight_error": 0.0,
}


def build_gemini_physical_logic_score_messages(sample: QASample, base64_images: list[str]) -> list[dict[str, Any]]:
    source_context = json.dumps(sample.source_context or sample.llm_fields, ensure_ascii=False, indent=2)
    user_text = "\n\n".join(
        [
            GEMINI_PHYSICAL_LOGIC_SCORE_RUBRIC.strip(),
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
        {"role": "system", "content": GEMINI_PHYSICAL_LOGIC_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Ordered visual evidence follows."},
                *image_blocks(base64_images),
                {"type": "text", "text": user_text},
            ],
        },
    ]


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


def parse_gemini_physical_logic_score_response(raw_text: str, *, task_name: str = "") -> dict[str, Any]:
    warnings: list[str] = []
    try:
        parsed = _extract_json_object(raw_text)
        scores = {
            "precondition_validity_score": _score(parsed.get("precondition_validity_score")),
            "causal_dependency_score": _score(parsed.get("causal_dependency_score")),
            "state_transition_score": _score(parsed.get("state_transition_score")),
            "timeline_consistency_score": _score(parsed.get("timeline_consistency_score")),
            "physical_feasibility_score": _score(parsed.get("physical_feasibility_score")),
        }
        rationales = {
            "precondition_rationale": str(parsed.get("precondition_rationale") or "").strip(),
            "causal_dependency_rationale": str(parsed.get("causal_dependency_rationale") or "").strip(),
            "state_transition_rationale": str(parsed.get("state_transition_rationale") or "").strip(),
            "timeline_rationale": str(parsed.get("timeline_rationale") or "").strip(),
            "physical_feasibility_rationale": str(parsed.get("physical_feasibility_rationale") or "").strip(),
        }
        hallucinated_raw = parsed.get("hallucinated_effects") or []
        if isinstance(hallucinated_raw, str):
            hallucinated_effects = [hallucinated_raw] if hallucinated_raw.strip() else []
        elif isinstance(hallucinated_raw, list):
            hallucinated_effects = [str(item).strip() for item in hallucinated_raw if str(item).strip()]
        else:
            hallucinated_effects = []
        tags = _tags(parsed.get("major_failure_tags") or parsed.get("failure_tags"))
        if hallucinated_effects and "hallucinated_effect" not in tags:
            tags = [tag for tag in tags if tag != "none"] + ["hallucinated_effect"]
        confidence = str(parsed.get("score_confidence") or "medium").strip().lower()
    except Exception as exc:
        return {
            "precondition_validity_score": 0.0,
            "causal_dependency_score": 0.0,
            "state_transition_score": 0.0,
            "timeline_consistency_score": 0.0,
            "physical_feasibility_score": 0.0,
            "overall_physical_logic_score": 0.0,
            "hallucinated_effects": [],
            "score_confidence": "low",
            "major_failure_tags": ["parse_error"],
            "score_formula": GEMINI_SCORE_FORMULA,
            "score_warnings": [f"parse_error:{type(exc).__name__}:{exc}"],
            "parse_error": True,
            "task_name": task_name,
        }

    values = list(scores.values())
    mean_score = round(sum(values) / len(values), 3)
    min_axis = min(values)
    caps = [mean_score, min_axis + 15.0]
    if min_axis < 25:
        caps.append(34.0)
        warnings.append("low_axis_cap:min_axis<25<=34")
    elif min_axis < 50:
        caps.append(64.0)
        warnings.append("low_axis_cap:min_axis<50<=64")
    for tag in [tag for tag in tags if tag != "none"]:
        if tag in GEMINI_HARD_FAILURE_CAPS:
            caps.append(GEMINI_HARD_FAILURE_CAPS[tag])
            warnings.append(f"hard_cap:{tag}<={GEMINI_HARD_FAILURE_CAPS[tag]}")
    if any(not value for value in rationales.values()):
        caps.append(60.0)
        warnings.append("missing_rationale_cap<=60")
    overall = round(max(0.0, min(100.0, min(caps))), 3)

    return {
        **scores,
        "overall_physical_logic_score": overall,
        **rationales,
        "hallucinated_effects": hallucinated_effects,
        "score_confidence": confidence if confidence in {"low", "medium", "high"} else "medium",
        "major_failure_tags": tags,
        "score_formula": GEMINI_SCORE_FORMULA,
        "score_warnings": warnings,
        "parse_error": False,
        "task_name": task_name,
    }
