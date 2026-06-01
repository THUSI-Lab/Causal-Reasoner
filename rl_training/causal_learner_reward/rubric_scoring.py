

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

try:
    from .config import DEFAULT_FAILURE_SCORE
    from .judge_prompts import normalize_task_id
except ImportError:
    from config import DEFAULT_FAILURE_SCORE                
    from judge_prompts import normalize_task_id                


ANSWER_VALUES = {
    "Yes": 1.0,
    "Partially": 0.5,
    "No": 0.0,
}

TASK_ANSWER_CHOICES: Dict[str, Dict[str, Sequence[str]]] = {
    "Task_01": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
    "Task_02": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
    "Task_06": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
    "Task_07": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
    "Task_18": {f"q{i}": ("Yes", "No") for i in range(1, 7)},
    "Task_19": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
    "Task_20": {f"q{i}": ("Yes", "Partially", "No") for i in range(1, 6)},
}

DIAGNOSTIC_KEYS: Dict[str, Sequence[str]] = {
    "Task_01": ("d1", "d2", "d3", "d4", "d5"),
    "Task_02": ("d1", "d2", "d3", "d4", "d5"),
    "Task_06": ("d1", "d2", "d3", "d4", "d5"),
    "Task_07": ("d1", "d2", "d3", "d4", "d5"),
    "Task_18": ("d1", "d2", "d3", "d4", "d5", "d6"),
    "Task_19": ("d1", "d2", "d3", "d4", "d5"),
    "Task_20": ("d1", "d2", "d3", "d4", "d5"),
}

DIAGNOSTIC_CAPS = {
    "d1": 0.35,
    "d2": 0.60,
    "d3": 0.25,
    "d4": 0.50,
    "d5": 0.45,
    "d6": 0.50,
}


@dataclass
class RubricScore:
    score: float
    valid: bool
    status: str
    invalid_reason: str | None
    answers: Dict[str, str]
    diagnostics: Dict[str, str]
    evidence: Dict[str, Any]


def score_judge_response(task_id: str, payload: Mapping[str, Any]) -> RubricScore:
    normalized = normalize_task_id(task_id)
    if payload.get("valid") is False:
        return RubricScore(
            score=DEFAULT_FAILURE_SCORE,
            valid=False,
            status="invalid_judge_input",
            invalid_reason=str(payload.get("invalid_reason") or "invalid_judge_input"),
            answers={},
            diagnostics={},
            evidence=_dict(payload.get("evidence")),
        )

    answer_choices = TASK_ANSWER_CHOICES[normalized]
    answers = _dict(payload.get("answers"))
    diagnostics = _dict(payload.get("diagnostics"))
    evidence = _dict(payload.get("evidence"))

    missing = [key for key in answer_choices if key not in answers]
    if missing:
        raise ValueError(f"judge_response_missing_answers: {missing}")

    values = []
    normalized_answers: Dict[str, str] = {}
    for key, allowed in answer_choices.items():
        value = _normalize_choice(answers.get(key))
        if value not in allowed:
            raise ValueError(f"invalid_answer_choice: {key}={answers.get(key)!r}, allowed={allowed}")
        normalized_answers[key] = value
        values.append(ANSWER_VALUES[value])

    normalized_diagnostics: Dict[str, str] = {}
    missing_diagnostics = [key for key in DIAGNOSTIC_KEYS[normalized] if key not in diagnostics]
    if missing_diagnostics:
        raise ValueError(f"judge_response_missing_diagnostics: {missing_diagnostics}")
    for key in DIAGNOSTIC_KEYS[normalized]:
        value = _normalize_choice(diagnostics.get(key))
        if value not in {"Yes", "No"}:
            raise ValueError(f"invalid_diagnostic_choice: {key}={diagnostics.get(key)!r}")
        normalized_diagnostics[key] = value

    missing_evidence = [key for key in answer_choices if not str(evidence.get(key) or "").strip()]
    if missing_evidence:
        raise ValueError(f"judge_response_missing_evidence: {missing_evidence}")

    score = sum(values) / len(values) if values else 0.0
    for key, value in normalized_diagnostics.items():
        if value == "Yes":
            score = min(score, DIAGNOSTIC_CAPS.get(key, 1.0))

    return RubricScore(
        score=max(0.0, min(1.0, score)),
        valid=True,
        status="ok",
        invalid_reason=None,
        answers=normalized_answers,
        diagnostics=normalized_diagnostics,
        evidence=evidence,
    )


def _normalize_choice(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "yes":
        return "Yes"
    if text == "partially":
        return "Partially"
    if text == "no":
        return "No"
    return str(value or "").strip()


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
