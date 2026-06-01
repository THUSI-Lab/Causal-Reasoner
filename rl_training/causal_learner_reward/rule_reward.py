

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

try:
    from .config import (
        DEFAULT_FACT_REWARD_WEIGHTS,
        DEFAULT_TASK18_WEIGHTS,
        DEFAULT_TASK20_WEIGHTS,
        RULE_DISABLED_TASKS,
    )
    from .data_schema import parse_ground_truth
except ImportError:                                                     
    from config import (                
        DEFAULT_FACT_REWARD_WEIGHTS,
        DEFAULT_TASK18_WEIGHTS,
        DEFAULT_TASK20_WEIGHTS,
        RULE_DISABLED_TASKS,
    )
    from data_schema import parse_ground_truth                


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "by",
    "can",
    "could",
    "do",
    "does",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "while",
    "will",
    "with",
    "would",
}

_ACTION_TERMS = {
    "adjust",
    "align",
    "clear",
    "close",
    "continue",
    "hold",
    "insert",
    "lift",
    "lower",
    "move",
    "open",
    "place",
    "press",
    "pull",
    "push",
    "reinsert",
    "re-lower",
    "remove",
    "repeat",
    "reposition",
    "restart",
    "secure",
    "separate",
    "spread",
    "stabilize",
    "turn",
    "widen",
}

_FLAW_TYPE_PATTERNS = {
    "goal_inconsistent": (
        "contradict",
        "wrong goal",
        "goal-inconsistent",
        "inconsistent with the goal",
        "does not advance",
        "irrelevant",
        "unrelated",
    ),
    "granularity_mismatch": (
        "too vague",
        "overly general",
        "granularity",
        "not specific",
        "lacks detail",
        "too broad",
    ),
    "precondition_missing": (
        "missing precondition",
        "without first",
        "before",
        "prerequisite",
        "required condition",
        "not yet",
    ),
    "redundant_step": (
        "redundant",
        "duplicate",
        "repeats",
        "unnecessary",
        "already done",
        "superfluous",
    ),
    "wrong_ordering": (
        "wrong order",
        "out of order",
        "sequence",
        "should occur before",
        "should occur after",
        "too early",
        "too late",
    ),
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: Any) -> List[str]:
    if text is None:
        return []
    raw = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", raw)
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _token_set(text: Any) -> Set[str]:
    return set(_tokens(text))


def _token_recall(pred: Any, gold: Any) -> float:
    gold_tokens = _token_set(gold)
    if not gold_tokens:
        return 0.0
    pred_tokens = _token_set(pred)
    return len(pred_tokens & gold_tokens) / len(gold_tokens)


def _token_precision(pred: Any, gold: Any) -> float:
    pred_tokens = _token_set(pred)
    if not pred_tokens:
        return 0.0
    gold_tokens = _token_set(gold)
    return len(pred_tokens & gold_tokens) / len(pred_tokens)


def _token_f1(pred: Any, gold: Any) -> float:
    precision = _token_precision(pred, gold)
    recall = _token_recall(pred, gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def compute_rule_reward(
    task_id: str,
    pred: Any,
    gold: Any,
    extra_info: Mapping[str, Any] | None = None,
    **config: Any,
) -> Dict[str, Any]:


    del extra_info
    gold_data = parse_ground_truth(gold)
    fields = gold_data.get("reward_fields") or {}
    response = str(pred or "").strip()

    if not response:
        return {"score": 0.0, "empty_output": True}

    if task_id in RULE_DISABLED_TASKS:
        return {"score": 0.0, "rule_disabled": True}
    if task_id in {"Task_01", "Task_02", "Task_06", "Task_07"}:
        return _rule_fact_task(response, fields, config.get("fact_reward_weights") or DEFAULT_FACT_REWARD_WEIGHTS)
    if task_id == "Task_18":
        return _rule_task18(response, fields, config.get("task18_weights") or DEFAULT_TASK18_WEIGHTS)
    if task_id == "Task_20":
        return _rule_task20(response, fields, config.get("task20_weights") or DEFAULT_TASK20_WEIGHTS)

    raise KeyError(f"Unsupported task_id for QA reward: {task_id}")


def _rule_fact_task(response: str, fields: Mapping[str, Any], weights: Mapping[str, float]) -> Dict[str, Any]:
    target_facts = _as_text_list(fields.get("target_facts"))
    target_text = str(fields.get("target_text") or "")
    patient = str(fields.get("patient") or "")

    fact_recall = _avg(_token_recall(response, fact) for fact in target_facts)
    answer_similarity = _token_f1(response, target_text)
    patient_score = 1.0 if patient and _token_recall(response, patient) > 0 else 0.0

    score = (
        weights.get("fact_recall", 0.70) * fact_recall
        + weights.get("answer_similarity", 0.20) * answer_similarity
        + weights.get("patient", 0.10) * patient_score
    )
    return {
        "score": _clamp(score),
        "fact_recall": fact_recall,
        "answer_similarity": answer_similarity,
        "patient_match": patient_score,
    }


def _rule_task18(response: str, fields: Mapping[str, Any], weights: Mapping[str, float]) -> Dict[str, Any]:
    flaw_step = int(fields.get("flaw_step") or 0)
    flaw_type = str(fields.get("flaw_type") or "")
    bad_steps = _as_text_list(fields.get("bad_plan_steps"))
    repair_steps = _as_text_list(fields.get("repair_steps"))

    step_score = _detect_flaw_step(response, flaw_step)
    type_score = _detect_flaw_type(response, flaw_type)
    repair_coverage = _avg(_token_recall(response, step) for step in repair_steps)

    bad_overlap = _avg(_token_recall(response, step) for step in bad_steps)
    anti_copy = 1.0
    if bad_overlap > 0.65 and repair_coverage < 0.45:
        anti_copy = 0.25
    elif bad_overlap > 0.75 and "corrected" not in response.lower() and "repair" not in response.lower():
        anti_copy = 0.55

    score = (
        weights.get("flaw_step", 0.15) * step_score
        + weights.get("flaw_type", 0.20) * type_score
        + weights.get("repair_coverage", 0.50) * repair_coverage
        + weights.get("anti_copy", 0.15) * anti_copy
    )
    return {
        "score": _clamp(score),
        "flaw_step_score": step_score,
        "flaw_type_score": type_score,
        "repair_coverage": repair_coverage,
        "bad_plan_overlap": bad_overlap,
        "anti_copy": anti_copy,
    }


def _detect_flaw_step(response: str, gold_step: int) -> float:
    if gold_step <= 0:
        return 0.0
    lowered = response.lower()
    direct_patterns = [
        rf"\bstep\s*{gold_step}\b",
        rf"\b{gold_step}\s*(?:st|nd|rd|th)?\s+step\b",
    ]
    if any(re.search(pattern, lowered) for pattern in direct_patterns):
        return 1.0
    numbers = [int(num) for num in re.findall(r"\bstep\s*(\d+)\b", lowered)]
    if not numbers:
        return 0.0
    if any(abs(num - gold_step) == 1 for num in numbers):
        return 0.5
    return 0.0


def _detect_flaw_type(response: str, flaw_type: str) -> float:
    if not flaw_type:
        return 0.0
    normalized = flaw_type.strip().lower()
    if normalized in response.lower():
        return 1.0
    patterns = _FLAW_TYPE_PATTERNS.get(normalized, ())
    if _contains_any(response, patterns):
        return 1.0
    type_words = normalized.replace("_", " ")
    return _token_f1(response, type_words)


def _rule_task20(response: str, fields: Mapping[str, Any], weights: Mapping[str, float]) -> Dict[str, Any]:
    failure_reason = str(fields.get("failure_reason") or "")
    recovery_strategy = str(fields.get("recovery_strategy") or "")
    step_goal = str(fields.get("step_goal") or "")

    recovery_coverage = _token_recall(response, recovery_strategy)
    failure_recall = _token_recall(response, failure_reason)
    action_signal = _action_signal(response, recovery_strategy, step_goal)
    novelty = _novelty_against_failure(response, failure_reason)
    failure_addressing = 1.0 if recovery_coverage >= 0.35 and failure_recall > 0 else recovery_coverage

    raw_score = (
        weights.get("recovery_coverage", 0.45) * recovery_coverage
        + weights.get("action_signal", 0.20) * action_signal
        + weights.get("novelty", 0.20) * novelty
        + weights.get("failure_addressing", 0.15) * failure_addressing
    )

    parrot_cap = 1.0
    if failure_recall >= 0.45 and recovery_coverage < 0.25:
        parrot_cap = 0.30
    elif failure_recall >= 0.65 and recovery_coverage < 0.45:
        parrot_cap = 0.55

    score = min(raw_score, parrot_cap)
    return {
        "score": _clamp(score),
        "recovery_coverage": recovery_coverage,
        "failure_reason_recall": failure_recall,
        "action_signal": action_signal,
        "novelty": novelty,
        "failure_addressing": failure_addressing,
        "parrot_cap": parrot_cap,
    }


def _action_signal(response: str, recovery_strategy: str, step_goal: str) -> float:
    response_tokens = _token_set(response)
    action_hit = 1.0 if response_tokens & _ACTION_TERMS else 0.0
    object_overlap = max(_token_recall(response, recovery_strategy), _token_recall(response, step_goal))
    return _clamp(0.45 * action_hit + 0.55 * object_overlap)


def _novelty_against_failure(response: str, failure_reason: str) -> float:
    response_tokens = _token_set(response)
    if not response_tokens:
        return 0.0
    failure_tokens = _token_set(failure_reason)
    novel_tokens = response_tokens - failure_tokens
    return len(novel_tokens) / len(response_tokens)
