from __future__ import annotations


CAUSAL_TRACE_SYSTEM_PROMPT = """\
You generate causal reasoning traces for embodied multimodal QA.
The trace must explain why the target answer follows from the question, visual evidence metadata, and source context.
Ground claims in the supplied inputs. Do not introduce unsupported objects, steps, states, or outcomes.
Separate visible or stated facts from inference, and connect them through physical, temporal, spatial, and procedural logic.
Return only the trace text. Do not include markdown fences, labels, or the final answer.
"""

TASK_TRACE_REQUIREMENTS = {
    "default": "Explain the decisive visual/source facts, causal mechanism, and answer justification.",
    "Task_18_Bad_Plan_Diagnosis_And_Repair": "Identify why the proposed step is invalid and how the repaired sequence restores plan coherence.",
    "Task_19_Counterfactual_Outcome": "Explain the counterfactual condition, blocked mechanism, and direct physical outcome.",
    "Task_20_Failure_Recovery": "Explain the failure cause, violated precondition, and recovery action that re-establishes executability.",
}


def task_requirement(task_name: str) -> str:
    return TASK_TRACE_REQUIREMENTS.get(task_name, TASK_TRACE_REQUIREMENTS["default"])
