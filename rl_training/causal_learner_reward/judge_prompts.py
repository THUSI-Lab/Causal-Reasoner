from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Mapping, Tuple


SYSTEM_PROMPT = (
    "You are a rigorous multimodal reward judge for embodied planning QA. Evaluate only the MODEL OUTPUT against the "
    "attached video, the QUESTION, and the GROUND TRUTH reference fields. Do not solve the task yourself, do not infer "
    "unstated facts, and do not reward plausible but visually unsupported content. Return one valid JSON object only."
)

COMMON_PROTOCOL = (
    "Evaluation protocol:\n"
    "1. Treat the attached video or sampled frames as the source of visual grounding and the ground truth as the task-specific reference target.\n"
    "2. If the model output conflicts with visible evidence, mark the affected rubric keys down even when the wording resembles the reference.\n"
    "3. Judge semantic correctness, visual grounding, physical feasibility, causal adequacy, temporal ordering, object binding, and task specificity independently.\n"
    "4. Give credit for equivalent wording only when the required object, relation, state, timing, polarity, and causal link are preserved.\n"
    "5. Do not give extra credit for fluency, length, confidence, or generic common sense. These cannot compensate for wrong grounding or wrong causality.\n"
    "6. Use Yes only for a substantively correct and grounded answer. Use Partially only for a directionally correct answer with limited omissions or mild vagueness. Use No for missing, reversed, unsupported, wrong-object, wrong-time, or wrong-task content.\n"
    "7. Penalize copied, generic, over-broad, hallucinated, temporally wrong, or physically impossible output even if it sounds fluent.\n"
    "8. Use only the allowed choices for every rubric and diagnostic key. Do not output numeric scores.\n"
    "9. Mark valid=false only if the judge input is structurally impossible to evaluate, such as missing question, missing model output, or unavailable visual evidence.\n"
    "10. The evidence object must include one concise grounded reason for every rubric key q1, q2, and so on. Each reason should mention the concrete object, action, state, relation, or causal link that determined the judgment.\n"
)

DIAGNOSTIC_TEXT = (
    "Diagnostic questions:\n"
    "D1. Is the model output mostly copied from the question or reference without actually answering?\n"
    "D2. Does the model output hallucinate visual details, objects, states, temporal order, affordances, or causal effects not supported by the video/reference?\n"
    "D3. Is the model output answering the wrong task type or optimizing a different objective?\n"
    "D4. Is the model output generic rather than grounded in this sample's concrete objects, actions, and state changes?\n"
    "D5. Does the model output include a physically impossible action, state, transition, mechanism, or recovery?\n"
)

TASK_SPECS = {
    "Task_01": {
        "title": "Spatial Precondition",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output cover the required spatial precondition facts that must hold before the step?",
            "Q2. Does the output bind each relevant object to the correct position, containment, contact, support, orientation, or relative spatial relation?",
            "Q3. Does the output state these facts as preconditions before the action, not as outcomes, goals, or unrelated states?",
            "Q4. Does the output preserve required contrast, absence, negation, or relational polarity when the reference depends on it?",
            "Q5. Does the output avoid unsupported spatial claims and avoid adding irrelevant locations or objects?",
        ],
        "choice_guidance": "Yes means all core spatial preconditions are complete, grounded, and temporally before the step. Partially means the answer direction is right but one important fact, binding, polarity, or temporal qualifier is missing or vague. No means the answer misses, reverses, mis-times, or fabricates the required precondition.",
    },
    "Task_02": {
        "title": "Affordance Precondition",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output cover the required affordance or object-state preconditions for executing the step?",
            "Q2. Does the output bind each affordance, material property, or enabling state to the correct object or object part?",
            "Q3. Does the output explain how the stated physical state enables the intended action?",
            "Q4. Does the output preserve important physical distinctions such as rigid vs flexible, open vs closed, attached vs loose, clean vs dirty, or available vs blocked?",
            "Q5. Does the output avoid unsupported affordance, material, tool-use, or property claims?",
        ],
        "choice_guidance": "Yes means all core affordance facts, object bindings, and enabling mechanisms are correct and visually grounded. Partially means the answer is directionally correct but underspecified, weakly causal, or missing one enabling detail. No means key affordance facts are absent, attached to the wrong object, temporally wrong, or physically unsupported.",
    },
    "Task_06": {
        "title": "Spatial Postcondition",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output describe the required final spatial state after the step?",
            "Q2. Does the output preserve the correct old-to-new movement, direction, placement, or containment change when required?",
            "Q3. Does the output cover the relevant objects, surfaces, containers, and locations involved in the final state?",
            "Q4. Does the output bind each final position or relation to the correct object or location?",
            "Q5. Does the output avoid unsupported postcondition spatial claims and avoid describing preconditions as final outcomes?",
        ],
        "choice_guidance": "Yes means the final spatial state, movement direction, and object-location bindings match the video/reference. Partially means one important object, location, direction, or relation is vague or omitted while the main postcondition remains correct. No means the final state is absent, wrong, precondition-like, wrong-object, or hallucinated.",
    },
    "Task_07": {
        "title": "Affordance Postcondition",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output describe the required affordance, material, or object-state change after the step?",
            "Q2. Does the output preserve the correct polarity of the change, such as opened vs closed, attached vs detached, cleaned vs dirty, or available vs unavailable?",
            "Q3. Does the output cover the relevant objects and the properties that changed?",
            "Q4. Does the output bind each changed property to the correct object or object part?",
            "Q5. Does the output avoid unsupported affordance or state-change claims and avoid treating unchanged properties as changed?",
        ],
        "choice_guidance": "Yes means the changed affordance/state, polarity, and object-property bindings are correct and grounded after the step. Partially means the change is mostly correct but incomplete, weakly specified, or missing one relevant property. No means the change is missing, reversed, assigned to the wrong object, described at the wrong time, or unsupported.",
    },
    "Task_18": {
        "title": "Bad Plan Diagnosis And Repair",
        "answers": {"q1": ("Yes", "No"), "q2": ("Yes", "No"), "q3": ("Yes", "No"), "q4": ("Yes", "No"), "q5": ("Yes", "No"), "q6": ("Yes", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5", "d6"),
        "rubric": [
            "Q1. Does the output explicitly identify that the proposed plan contains a flaw?",
            "Q2. Does the output localize the correct flawed step or segment of the plan?",
            "Q3. Does the output explain the flaw type using the high-level goal, prefix context, and physical preconditions shown in the video?",
            "Q4. Does the output provide a corrected plan or repair rather than only criticizing the bad plan?",
            "Q5. Does the repair cover the key reference repair steps in the correct causal/temporal order?",
            "Q6. Is the repaired plan physically feasible in the shown scene and consistent with available objects and states?",
        ],
        "choice_guidance": "Use Yes only when the criterion is substantively satisfied with the correct flawed segment, causal diagnosis, repair content, order, and physical feasibility. Use No for missing, wrong, vague, non-localized, copied, unsupported, out-of-order, or physically infeasible diagnosis/repair.",
        "extra_diagnostic": "D6. Does the model output fail to provide a corrected plan?",
    },
    "Task_19": {
        "title": "Counterfactual Outcome",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output use the counterfactual condition stated in the question/reference rather than answering the factual case?",
            "Q2. Does the output state the specific expected outcome under that counterfactual condition?",
            "Q3. Does the output give a physically and causally valid link from the condition to the predicted outcome?",
            "Q4. Is the output more than a restatement of the question, with an actual predicted outcome and reason?",
            "Q5. Does the output avoid giving recovery advice or normative instructions instead of predicting the counterfactual outcome?",
        ],
        "choice_guidance": "Yes means the counterfactual condition, predicted outcome, and causal mechanism are clear, grounded, and not confused with the factual video outcome. Partially means the answer is directionally right but incomplete, weakly causal, or underspecified. No means it ignores the condition, predicts the wrong outcome, gives advice instead, or changes task type.",
    },
    "Task_20": {
        "title": "Failure Recovery",
        "answers": {"q1": ("Yes", "Partially", "No"), "q2": ("Yes", "Partially", "No"), "q3": ("Yes", "Partially", "No"), "q4": ("Yes", "Partially", "No"), "q5": ("Yes", "Partially", "No")},
        "diagnostics": ("d1", "d2", "d3", "d4", "d5"),
        "rubric": [
            "Q1. Does the output propose a concrete, executable recovery action?",
            "Q2. Does the output bind the recovery action to the correct object, tool, target state, and method?",
            "Q3. Is the proposed action causally adequate for recovering from the stated failure?",
            "Q4. Does the output address the root cause of the failure rather than merely repeating the failure description?",
            "Q5. Is the recovery physically feasible in the shown scene and not merely copied from the prompt/reference?",
        ],
        "choice_guidance": "Yes means the recovery action is executable, grounded, causally adequate, root-cause-directed, and object-specific. Partially means it may help but is vague, incomplete, weakly tied to the root cause, or missing one object/method detail. No means no actionable recovery is given, the wrong object/method is used, the root cause is ignored, or the recovery is infeasible.",
    },
}


@lru_cache(maxsize=None)
def load_runtime_prompt(task_id: str) -> str:
    normalized = normalize_task_id(task_id)
    spec = TASK_SPECS[normalized]
    answer_lines = []
    for key, allowed in spec["answers"].items():
        answer_lines.append(f"{key}: allowed choices are {' / '.join(allowed)}")
    diagnostic_lines = [DIAGNOSTIC_TEXT]
    if spec.get("extra_diagnostic"):
        diagnostic_lines.append(str(spec["extra_diagnostic"]))
    diagnostic_lines.append("Allowed diagnostic answers: Yes / No only.")
    evidence_keys = ", ".join(spec["answers"])
    schema = {
        "task_id": normalized.lower(),
        "valid": True,
        "invalid_reason": None,
        "answers": {key: "<allowed choice>" for key in spec["answers"]},
        "diagnostics": {key: "<Yes or No>" for key in spec["diagnostics"]},
        "evidence": {key: "<short grounded reason>" for key in spec["answers"]},
    }
    return (
        f"Task: {normalized} {spec['title']}\n\n"
        + COMMON_PROTOCOL
        + "\nRubric questions:\n"
        + "\n".join(spec["rubric"])
        + "\n\nAllowed answer choices:\n"
        + "\n".join(answer_lines)
        + "\n\nChoice guidance:\n"
        + str(spec["choice_guidance"])
        + "\n\n"
        + "\n".join(diagnostic_lines)
        + "\n\nOutput schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + f"\n\nReturn all answer keys, all diagnostic keys, and evidence for {evidence_keys}. Do not include prose outside JSON. Return JSON only."
    )


def build_judge_messages(
    *,
    task_id: str,
    question: str,
    ground_truth: Mapping[str, Any],
    model_output: str,
) -> Tuple[str, str]:
    normalized = normalize_task_id(task_id)
    prompt = load_runtime_prompt(normalized)
    user_text = (
        prompt
        + "\n\nJudge input:\n"
        + f"Task id: {normalized}\n"
        + "Attached visual evidence: the video or sampled frames supplied with this message.\n\n"
        + f"Question:\n{question}\n\n"
        + "Ground truth / reference fields:\n"
        + json.dumps(ground_truth, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nModel output:\n"
        + (model_output or "")
        + "\n\nEvaluate the model output against the rubric using strict visual and causal grounding. Return JSON only."
    )
    return SYSTEM_PROMPT, user_text


def normalize_task_id(task_id: str) -> str:
    text = str(task_id or "").strip()
    match = re.match(r"(?i)^task_(\d{1,2})(?:_|$)", text)
    if match:
        return f"Task_{int(match.group(1)):02d}"
    raise KeyError(f"Unsupported task_id: {task_id}")
