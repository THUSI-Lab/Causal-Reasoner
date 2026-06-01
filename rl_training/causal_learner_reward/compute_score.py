

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from .config import DEFAULT_ALPHA_TABLE, DEFAULT_FAILURE_SCORE, DEFAULT_RUBRIC_ID, JudgeAPIConfig
    from .data_schema import SCHEMA_ID, parse_ground_truth
    from .judge_client import JudgeRequest, get_judge_client, reset_judge_clients
    from .rubric_scoring import score_judge_response
    from .rule_reward import compute_rule_reward
except ImportError:
    from config import DEFAULT_ALPHA_TABLE, DEFAULT_FAILURE_SCORE, DEFAULT_RUBRIC_ID, JudgeAPIConfig                
    from data_schema import SCHEMA_ID, parse_ground_truth                
    from judge_client import JudgeRequest, get_judge_client, reset_judge_clients                
    from rubric_scoring import score_judge_response                
    from rule_reward import compute_rule_reward                


logger = logging.getLogger(__name__)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    phase: int = 1,
    alpha_table: Optional[Dict[str, float]] = None,
    judge_api_config: Optional[Dict[str, Any]] = None,
    rubric_id: str = DEFAULT_RUBRIC_ID,
    **kwargs: Any,
) -> Dict[str, Any]:


    return compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info or {}],
        phase=phase,
        alpha_table=alpha_table,
        judge_api_config=judge_api_config,
        rubric_id=rubric_id,
        **kwargs,
    )[0]


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Dict[str, Any]]] = None,
    phase: int = 1,
    alpha_table: Optional[Dict[str, float]] = None,
    judge_api_config: Optional[Dict[str, Any]] = None,
    rubric_id: str = DEFAULT_RUBRIC_ID,
    **kwargs: Any,
) -> List[Dict[str, Any]]:


    del data_sources
    alpha_table = alpha_table or DEFAULT_ALPHA_TABLE
    extra_infos = extra_infos or [{} for _ in solution_strs]
    if not (len(solution_strs) == len(ground_truths) == len(extra_infos)):
        raise ValueError("solution_strs, ground_truths, and extra_infos must have the same length")

    prepared: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], str]] = []
    pending_indices: List[int] = []
    judge_requests: List[JudgeRequest] = []
    results: List[Dict[str, Any]] = []

    judge_config = None
    if phase >= 2:
        try:
            judge_config = JudgeAPIConfig.from_mapping(judge_api_config)
            if judge_config is None:
                raise ValueError("missing_judge_api_config")
            judge_config.validate()
        except Exception as exc:
            return [
                _failure_result(_safe_task_id(gt, extra), f"judge_config_error: {exc}", phase=phase)
                for gt, extra in zip(ground_truths, extra_infos)
            ]

    for response_raw, ground_truth, extra_info_raw in zip(solution_strs, ground_truths, extra_infos):
        extra_info = _ensure_dict(extra_info_raw)
        response = (response_raw or "").strip()
        base_result, gold, extra_info, task_id = _compute_rule_part(
            response=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
            phase=phase,
            alpha_table=alpha_table,
            **kwargs,
        )
        results.append(base_result)
        prepared.append((base_result, gold, extra_info, task_id))

        if phase >= 2 and gold is not None and response and judge_config is not None:
            pending_indices.append(len(results) - 1)
            judge_requests.append(
                JudgeRequest(
                    task_id=task_id,
                    question=str(extra_info.get("question") or ""),
                    ground_truth=gold,
                    model_output=response,
                    extra_info=extra_info,
                    rubric_id=rubric_id,
                )
            )

    if not judge_requests:
        return results

    judge_client = get_judge_client(judge_config)
    judge_results = judge_client.judge_many_sync(judge_requests)
    if len(judge_results) != len(judge_requests):
        raise RuntimeError(
            f"judge_result_count_mismatch: expected {len(judge_requests)}, got {len(judge_results)}"
        )
    for result_index, judge_result in zip(pending_indices, judge_results):
        base_result, _gold, _extra_info, task_id = prepared[result_index]
        if judge_result.status != "ok" or judge_result.payload is None:
            failed = _failure_result(
                task_id,
                f"{judge_result.status}: {judge_result.error or 'judge_failed'}",
                phase=phase,
                r_rule=base_result.get("r_rule", 0.0),
                alpha=base_result.get("alpha", 0.0),
            )
            failed.update(_judge_result_metrics(judge_result, judge_invalid=1))
            results[result_index] = failed
            continue

        try:
            rubric = score_judge_response(task_id, judge_result.payload)
        except Exception as exc:
            failed = _failure_result(
                task_id,
                f"judge_response_parse_error: {exc}",
                phase=phase,
                r_rule=base_result.get("r_rule", 0.0),
                alpha=base_result.get("alpha", 0.0),
            )
            failed.update(_judge_result_metrics(judge_result, judge_invalid=1))
            results[result_index] = failed
            continue

        if not rubric.valid:
            failed = _failure_result(
                task_id,
                f"invalid_judge_input: {rubric.invalid_reason}",
                phase=phase,
                r_rule=base_result.get("r_rule", 0.0),
                alpha=base_result.get("alpha", 0.0),
            )
            failed.update(_judge_result_metrics(judge_result, judge_invalid=1))
            results[result_index] = failed
            continue

        alpha = float(base_result.get("alpha", 0.0))
        r_rule = float(base_result.get("r_rule", 0.0))
        r_judge = rubric.score
        final_score = alpha * r_rule + (1.0 - alpha) * r_judge
        base_result.update(
            {
                "score": _clamp(final_score),
                "r_judge": _clamp(r_judge),
                "judge_status": "ok",
                "judge_cached": judge_result.cached,
                "judge_video_key": judge_result.video_key,
                "judge_answers": rubric.answers,
                "judge_diagnostics": rubric.diagnostics,
                "judge_evidence": rubric.evidence,
                "rubric_id": rubric_id,
            }
        )
        base_result.update(_judge_result_metrics(judge_result, judge_invalid=0))
        results[result_index] = base_result

    return results


def _compute_rule_part(
    *,
    response: str,
    ground_truth: Any,
    extra_info: Dict[str, Any],
    phase: int,
    alpha_table: Dict[str, float],
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], str]:
    try:
        gold = parse_ground_truth(ground_truth)
    except Exception as exc:
        logger.error("failed to parse ground_truth: %s", exc, exc_info=True)
        task_id = str(extra_info.get("task_id", ""))
        return _failure_result(task_id, f"ground_truth_parse_error: {exc}", phase=phase), None, extra_info, task_id

    task_id = gold.get("task_id") or extra_info.get("task_id") or ""
    if not task_id:
        return _failure_result("", "missing_task_id", phase=phase), None, extra_info, ""
    if gold.get("schema_id") != SCHEMA_ID:
        return (
            _failure_result(task_id, f"schema_id_mismatch: {gold.get('schema_id')}", phase=phase),
            None,
            extra_info,
            task_id,
        )

    alpha = float(alpha_table.get(task_id, 0.0))
    if not response:
        return (
            {
                "score": 0.0,
                "task_id": task_id,
                "r_rule": 0.0,
                "r_judge": 0.0,
                "alpha": alpha,
                "phase": phase,
                "empty_output": True,
                "judge_status": "not_requested",
            },
            gold,
            extra_info,
            task_id,
        )

    try:
        rule_result = compute_rule_reward(task_id=task_id, pred=response, gold=gold, extra_info=extra_info, **kwargs)
    except KeyError as exc:
        logger.error("[%s] unsupported task: %s", task_id, exc)
        return _failure_result(task_id, f"rule_dispatch_error: {exc}", phase=phase), None, extra_info, task_id
    except Exception as exc:
        logger.error("[%s] rule reward failed: %s", task_id, exc, exc_info=True)
        return _failure_result(task_id, f"rule_computation_error: {exc}", phase=phase), None, extra_info, task_id

    r_rule = _clamp(float(rule_result.get("score", 0.0)))
    result = {
        "score": r_rule,
        "task_id": task_id,
        "r_rule": r_rule,
        "r_judge": 0.0,
        "alpha": alpha,
        "phase": phase,
        "judge_status": "not_requested" if phase < 2 else "pending",
    }
    for key, value in rule_result.items():
        if key == "score":
            continue
        result[f"rule_{key}"] = value
    return result, gold, extra_info, task_id


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_task_id(ground_truth: Any, extra_info: Any) -> str:
    try:
        gold = parse_ground_truth(ground_truth)
        return str(gold.get("task_id") or "")
    except Exception:
        return str(_ensure_dict(extra_info).get("task_id", ""))


def _failure_result(
    task_id: str,
    error: str,
    *,
    phase: int = 1,
    r_rule: float = 0.0,
    alpha: float = 0.0,
) -> Dict[str, Any]:
    return {
        "score": DEFAULT_FAILURE_SCORE,
        "task_id": task_id,
        "r_rule": _clamp(r_rule),
        "r_judge": 0.0,
        "alpha": float(alpha),
        "phase": phase,
        "judge_status": "failed" if phase >= 2 else "not_requested",
        "error": error,
    }


def _judge_result_metrics(judge_result: Any, *, judge_invalid: int) -> Dict[str, Any]:
    return {
        "judge_request_count": int(getattr(judge_result, "request_count", 0)),
        "judge_retry_count": int(getattr(judge_result, "retry_count", 0)),
        "judge_pool_timeout_count": int(getattr(judge_result, "pool_timeout_count", 0)),
        "judge_endpoint_timeout_count": int(getattr(judge_result, "endpoint_timeout_count", 0)),
        "judge_cache_hit": int(getattr(judge_result, "cache_hit", 0)),
        "judge_video_preprocess_count": int(getattr(judge_result, "video_preprocess_count", 0)),
        "judge_invalid": int(judge_invalid),
    }


def reset() -> None:
    reset_judge_clients()
