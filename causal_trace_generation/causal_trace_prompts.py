


from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List


TRACE_PIPELINE_ID = "causal_reasoning_traces"


TASK_NAMES = [
    "Task_01_Spatial_Precondition",
    "Task_02_Affordance_Precondition",
    "Task_03_Physical_Feasibility",
    "Task_04_Affordance_Visual_Semantics",
    "Task_05_Holistic_Causal_Chain",
    "Task_06_Spatial_Postcondition",
    "Task_07_Affordance_Postcondition",
    "Task_08_Goal_Recognition",
    "Task_09_Macro_Anchor_Extraction",
    "Task_10_Clip_to_Step_Goal",
    "Task_11_Action_Phrase",
    "Task_12_State_Evolution",
    "Task_13_Strategic_Rationale",
    "Task_14_Inter_Step_Dependency",
    "Task_15_Next_Step_Prediction",
    "Task_16_Middle_Steps_Infill",
    "Task_17_Next_K_Steps_Prediction",
    "Task_18_Bad_Plan_Diagnosis_And_Repair",
    "Task_19_Counterfactual_Outcome",
    "Task_20_Failure_Recovery",
]


_FAMILY_GUIDANCE = {
    "goal_and_perception": {
        "description": "Perception-to-intent traces for recognizing goals, anchor objects, step goals, and action phrases.",
        "global_moves": [
            "Start from visible objects, spatial layout, motion, and temporal order.",
            "Convert those observations into subgoals or action roles.",
            "Disambiguate at least one plausible alternative interpretation.",
            "End on the decisive causal observation, not by copying the answer.",
        ],
    },
    "physical_state_affordance": {
        "description": "Physical state and affordance traces for object properties, state transitions, preconditions, and postconditions.",
        "global_moves": [
            "Name the relevant object state before the action.",
            "Explain what force, contact, material property, geometry, or containment relation makes the transition possible.",
            "Trace the resulting spatial or affordance change.",
            "Include a counterfactual failure test for a missing property or mispositioned object.",
        ],
    },
    "causal_verification": {
        "description": "Explicit condition-checking traces for holistic chains, strategic necessity, feasibility, and inter-step dependency.",
        "global_moves": [
            "Separate spatial preconditions from affordance or functional preconditions.",
            "Trace the mechanism that links the checked conditions to the action.",
            "Verify the postconditions or downstream dependency created by the action.",
            "Use a skip/removal/failure test to prove necessity rather than merely describing correlation.",
        ],
    },
    "procedural_planning": {
        "description": "Procedural planning traces for next-step prediction, gap filling, multi-step forecasting, and plan repair.",
        "global_moves": [
            "Inventory the cumulative state left by completed steps.",
            "Compare that state against the next required preconditions.",
            "Construct or repair the step sequence by matching each postcondition to the next precondition.",
            "Reject at least one tempting but causally weaker ordering or action.",
        ],
    },
    "counterfactual_recovery": {
        "description": "Counterfactual and recovery traces for disrupted outcomes and repair protocols.",
        "global_moves": [
            "Reconstruct the normal causal chain first.",
            "Identify the exact broken link: spatial, affordance, force, material, or temporal.",
            "Propagate consequences through at least immediate and downstream effects.",
            "For recovery, state which preconditions must be restored and why the repair restores them.",
        ],
    },
}


def _clusters(*items: tuple[str, List[str]]) -> List[Dict[str, List[str]]]:
    return [{"name": name, "keywords": keywords} for name, keywords in items]


_BASE_CLUSTERS = _clusters(
    ("observation_grounding", ["visible", "observe", "position", "state", "motion", "contact", "object"]),
    ("causal_linking", ["because", "therefore", "requires", "enables", "prevents", "depends", "leads"]),
    ("mechanism", ["force", "motion", "contact", "friction", "gravity", "rigid", "surface", "trajectory"]),
    ("counterfactual", ["if", "without", "otherwise", "would", "alternative", "rules out", "failure"]),
)


TASK_TRACE_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "Task_08_Goal_Recognition": {
        "family": "goal_and_perception",
        "trace_goal": "Infer the high-level objective from the temporal chain of observed subgoals.",
        "required_moves": [
            "Catalog the major actions in order and name the manipulated objects.",
            "Explain what state each action creates and what next action it enables.",
            "Show the common objective that ties the subgoals together.",
            "Rule out a weaker interpretation such as random manipulation or mere cleanup.",
        ],
        "min_words": 210,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("temporal_goal", ["sequence", "subgoal", "overall", "goal", "objective", "then"]),
        ),
    },
    "Task_09_Macro_Anchor_Extraction": {
        "family": "goal_and_perception",
        "trace_goal": "Identify causally central objects by testing manipulation, state change, and necessity.",
        "required_moves": [
            "Evaluate each candidate object as tool, patient, container, support surface, or bystander.",
            "Use the removal test: what step fails if the object is absent.",
            "Distinguish directly manipulated anchors from background objects.",
            "Explain the causal role of every selected anchor without ending as a copied list.",
        ],
        "min_words": 200,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("anchor_role", ["tool", "patient", "container", "surface", "anchor", "removed", "bystander"]),
        ),
    },
    "Task_10_Clip_to_Step_Goal": {
        "family": "goal_and_perception",
        "trace_goal": "Infer the step goal from a local clip by connecting motion to state change and plan context.",
        "required_moves": [
            "Describe the visible motion, contact path, and acted-on object.",
            "Explain the state change produced by that motion.",
            "Connect the state change to what the broader plan needs next.",
            "Rule out a plausible alternate step goal.",
        ],
        "min_words": 210,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("step_goal", ["step", "goal", "state change", "prior", "next", "enable"]),
        ),
    },
    "Task_11_Action_Phrase": {
        "family": "goal_and_perception",
        "trace_goal": "Derive the precise action phrase from kinematics, agent-tool contact, and object response.",
        "required_moves": [
            "Analyze trajectory, speed, force direction, grip, and contact interface.",
            "Compare candidate verbs and state which visual feature selects the correct one.",
            "Explain how the patient object's response confirms the action.",
            "State why a different tool or motion would not produce the same effect.",
        ],
        "min_words": 220,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("kinematics", ["trajectory", "grip", "verb", "tool", "patient", "motion", "action"]),
        ),
    },
    "Task_04_Affordance_Visual_Semantics": {
        "family": "physical_state_affordance",
        "trace_goal": "Ground the hotspot affordance in object geometry, material, contact, and mechanism.",
        "required_moves": [
            "Name the acted-on object or hotspot and its relevant physical properties.",
            "Trace property to contact type, force application, and resulting interaction.",
            "Explain what would fail if the property were absent or altered.",
            "Use the exact affordance type as a causal role, not just a label.",
        ],
        "min_words": 210,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("affordance", ["affordance", "geometry", "material", "property", "hotspot", "interaction"]),
        ),
    },
    "Task_12_State_Evolution": {
        "family": "physical_state_affordance",
        "trace_goal": "Explain a state transition through before-state, force/action, physical principle, and after-state.",
        "required_moves": [
            "Inventory the before-state: position, orientation, configuration, and relevant property.",
            "Identify the force source and the contact path that drives the change.",
            "Name the physical principle that makes this change happen.",
            "Characterize the after-state and what it newly enables or blocks.",
        ],
        "min_words": 220,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("state_transition", ["before", "after", "state", "configuration", "transition", "changed"]),
        ),
    },
    "Task_05_Holistic_Causal_Chain": {
        "family": "causal_verification",
        "trace_goal": "Trace spatial preconditions, affordance preconditions, mechanism, spatial effects, and affordance effects.",
        "required_moves": [
            "Separate spatial preconditions from affordance preconditions.",
            "Name the agent, action, patient, and force/contact mechanism.",
            "Trace spatial effects and affordance effects after the action.",
            "Explain how the full chain supports the concise answer.",
        ],
        "min_words": 260,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("five_layer_chain", ["spatial", "precondition", "affordance", "mechanism", "effect", "postcondition"]),
        ),
    },
    "Task_13_Strategic_Rationale": {
        "family": "causal_verification",
        "trace_goal": "Prove why a step is strategically necessary by linking its postconditions to downstream preconditions.",
        "required_moves": [
            "Identify the postconditions created by the step.",
            "Name the later step or goal requirement that depends on those postconditions.",
            "Run a skip test and identify the first downstream failure.",
            "Explain why the step is a unique or most direct provider of the needed condition.",
        ],
        "min_words": 240,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("necessity", ["necessary", "skip", "downstream", "postcondition", "precondition", "fails"]),
        ),
    },
    "Task_01_Spatial_Precondition": {
        "family": "physical_state_affordance",
        "trace_goal": "Explain which spatial arrangements must hold before the action and why each is necessary.",
        "required_moves": [
            "Name the required object positions, orientations, alignments, and proximity relations.",
            "Tie each relation to reach, contact, trajectory, support, or containment.",
            "Describe what physical sub-motion would fail if one relation were absent.",
            "Avoid generic 'in the right place' wording.",
        ],
        "min_words": 210,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("spatial_precondition", ["spatial", "position", "orientation", "aligned", "proximity", "reach"]),
        ),
    },
    "Task_02_Affordance_Precondition": {
        "family": "physical_state_affordance",
        "trace_goal": "Explain which functional object states must hold before the action and why the action depends on them.",
        "required_moves": [
            "Name each required affordance state separately, such as open, graspable, rigid, empty, sharp, flexible, unobstructed, or accessible.",
            "For every named property, trace the exact sub-motion or force path that depends on it: where force is applied, what contact path carries it, and what would be blocked without the property.",
            "Run a concrete failure test for at least two missing properties; describe the failed physical motion, not only the missing label.",
            "State the provenance of each prerequisite when possible: whether it was already true before the clip, created by an earlier step, or maintained by the current spatial setup.",
            "End by tying the properties together as joint affordance readiness; avoid a mere list of object properties.",
        ],
        "min_words": 260,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("affordance_precondition", ["affordance", "functional", "property", "open", "grasp", "accessible", "rigid", "flexible"]),
            ("affordance_provenance", ["earlier", "already", "before", "prior", "maintained", "created"]),
            ("affordance_failure_test", ["if", "without", "would fail", "could not", "blocked", "jam", "collapse"]),
        ),
    },
    "Task_03_Physical_Feasibility": {
        "family": "causal_verification",
        "trace_goal": "Verify feasibility by checking spatial and affordance conditions jointly.",
        "required_moves": [
            "Check each spatial precondition and explain why the action requires it.",
            "Check each affordance precondition and explain the physical dependency.",
            "Combine the checks to derive the feasibility verdict.",
            "State what would fail if one checked condition were false.",
        ],
        "min_words": 260,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("feasibility", ["feasible", "spatial", "affordance", "condition", "joint", "verdict"]),
        ),
    },
    "Task_06_Spatial_Postcondition": {
        "family": "physical_state_affordance",
        "trace_goal": "Explain how the action changes spatial relationships and what the new layout enables.",
        "required_moves": [
            "Recall the initial spatial layout of the relevant objects.",
            "Trace displacement, rotation, transfer, containment, support, clearance, or alignment.",
            "Name the new spatial relationship and the next action it enables.",
            "Explain what would be blocked if this postcondition did not occur.",
        ],
        "min_words": 210,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("spatial_postcondition", ["after", "spatial", "moved", "position", "relationship", "enables"]),
        ),
    },
    "Task_07_Affordance_Postcondition": {
        "family": "physical_state_affordance",
        "trace_goal": "Explain how the action transforms functional properties and future capabilities.",
        "required_moves": [
            "Recall the before-action affordance state.",
            "Trace the physical cause of the affordance change.",
            "Name the new capability or limitation created by the transformation.",
            "Connect that new affordance state to a later plan step.",
        ],
        "min_words": 220,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("affordance_postcondition", ["after", "affordance", "functional", "capability", "transforms", "future"]),
        ),
    },
    "Task_14_Inter_Step_Dependency": {
        "family": "causal_verification",
        "trace_goal": "Explain how one step's effects satisfy the next step's preconditions.",
        "required_moves": [
            "List the prior step's spatial and affordance effects.",
            "List the following step's spatial and affordance preconditions.",
            "Match specific effects to specific required conditions.",
            "Run the removal test: if the first step were absent, which precondition would fail.",
        ],
        "min_words": 230,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("inter_step", ["previous", "next", "dependency", "precondition", "postcondition", "satisfies"]),
        ),
    },
    "Task_15_Next_Step_Prediction": {
        "family": "procedural_planning",
        "trace_goal": "Predict the next step by matching the current cumulative state to the next unmet goal condition.",
        "required_moves": [
            "Inventory the exact cumulative state at the prefix boundary: completed objects, unfinished objects, current locations, and functional states.",
            "Identify the remaining goal requirement and the newly satisfied preconditions that make the predicted next step possible now.",
            "Compare at least two candidate next actions: the chosen one and one tempting alternative. Explain which precondition or temporal dependency makes the alternative weaker.",
            "Explain why the chosen step is the earliest causally valid successor, not merely a plausible future action.",
            "Connect the chosen step's expected postcondition to a later downstream step in the plan.",
        ],
        "min_words": 270,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("next_step", ["prefix", "current", "next", "remaining", "candidate", "precondition"]),
            ("alternative_comparison", ["alternative", "candidate", "tempting", "weaker", "rather than", "instead"]),
            ("earliest_successor", ["earliest", "successor", "now", "remaining", "downstream", "later"]),
        ),
    },
    "Task_16_Middle_Steps_Infill": {
        "family": "procedural_planning",
        "trace_goal": "Infer missing middle steps by comparing head postconditions with tail preconditions.",
        "required_moves": [
            "State what the first observed step leaves true.",
            "State what the final observed step requires before it can begin.",
            "Identify each state gap that must be closed.",
            "Justify every inferred middle step as creating a required precondition for the next link.",
        ],
        "min_words": 260,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("gap_bridge", ["gap", "middle", "bridge", "head", "tail", "precondition", "postcondition"]),
        ),
    },
    "Task_17_Next_K_Steps_Prediction": {
        "family": "procedural_planning",
        "trace_goal": "Forecast a sequence by chaining each predicted step's postconditions into the next step's preconditions.",
        "required_moves": [
            "Describe the state after the prefix.",
            "For each predicted step, name the condition that makes it possible now.",
            "Explain what each step creates for the following step.",
            "Verify the predicted sequence moves toward the overall goal.",
        ],
        "min_words": 250,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("multi_step", ["sequence", "prefix", "predicted", "next", "postcondition", "toward"]),
        ),
    },
    "Task_18_Bad_Plan_Diagnosis_And_Repair": {
        "family": "procedural_planning",
        "trace_goal": "Diagnose a flawed plan by locating the broken causal link and explaining how the repair restores continuity.",
        "required_moves": [
            "Walk the plan in order and check each step's preconditions and postconditions.",
            "Identify the flawed step and flaw type.",
            "Explain the broken mechanism or missing state transition.",
            "Verify that the repaired plan restores the required causal order.",
        ],
        "min_words": 260,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("diagnosis_repair", ["flaw", "repair", "plan", "order", "broken", "restores"]),
        ),
    },
    "Task_19_Counterfactual_Outcome": {
        "family": "counterfactual_recovery",
        "trace_goal": "Predict the disrupted outcome by contrasting the normal causal chain with the counterfactual broken chain.",
        "required_moves": [
            "Reconstruct the normal chain from preconditions to mechanism to intended postconditions.",
            "Identify the exact counterfactual break point.",
            "Propagate the consequence through at least immediate, secondary, and final plan-level effects.",
            "Contrast the normal end-state with the degraded counterfactual end-state.",
        ],
        "min_words": 290,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("counterfactual_cascade", ["counterfactual", "normal", "disruption", "cascade", "downstream", "outcome"]),
        ),
    },
    "Task_20_Failure_Recovery": {
        "family": "counterfactual_recovery",
        "trace_goal": "Explain recovery by diagnosing the failure state and restoring the preconditions needed to resume.",
        "required_moves": [
            "Describe the failure configuration precisely.",
            "Diagnose the physical cause of the failure.",
            "Map each recovery action to the condition it restores.",
            "Verify the original step can resume after the recovery.",
        ],
        "min_words": 270,
        "validation_clusters": _BASE_CLUSTERS + _clusters(
            ("recovery", ["failure", "recover", "restore", "resume", "condition", "precondition"]),
        ),
    },
}


TASK_FAMILY_BY_NAME = {task: contract["family"] for task, contract in TASK_TRACE_CONTRACTS.items()}


_COMMON_PREAMBLE = """\
You are writing high-quality causal reasoning traces for embodied visual QA.
The trace will be placed inside <think>...</think> tags by the data pipeline,
but you must output only the trace text itself.

Core contract:
- Reason from grounded visual and structured evidence, not from the answer text alone.
- Treat structured fields and plan context as the fact boundary. Use them to build
  the causal skeleton; do not invent new objects, steps, outcomes, or hidden events.
- Keep the final answer separate. The answer is already known and will be appended
  outside the trace. Do not copy the full answer, do not end by restating it, and
  do not write "therefore the answer is".
- Make every important inference causal: condition -> mechanism -> effect -> next
  condition. Name the specific object, spatial relation, material property, contact
  path, force, affordance, or plan dependency that carries the link.
- Include at least one disambiguation or counterfactual/failure test unless the task
  contract gives a stronger task-specific alternative.
- Never mention JSON, metadata, prompts, paths, frame numbers, being an AI, or having
  been given an answer. Write as an expert observer reconstructing the reasoning.
- Output plain analytical prose only. No bullets, numbered lists, headings, XML tags,
  markdown, or meta-commentary.
"""


def _system_prompt_for(task_name: str) -> str:
    contract = TASK_TRACE_CONTRACTS[task_name]
    family = _FAMILY_GUIDANCE[contract["family"]]
    moves = "\n".join(f"- {move}" for move in family["global_moves"])
    required = "\n".join(f"- {move}" for move in contract["required_moves"])
    return (
        f"{_COMMON_PREAMBLE}\n"
        f"Task family: {contract['family']} - {family['description']}\n\n"
        f"Family-level reasoning moves:\n{moves}\n\n"
        f"Task-specific trace goal:\n{contract['trace_goal']}\n\n"
        f"Required task-specific moves:\n{required}\n\n"
        f"Minimum depth: at least {contract['min_words']} words unless the evidence is genuinely sparse. "
        "Short answers require deeper traces, because the trace must decompress the perceptual and causal work hidden by the answer."
    )


CAUSAL_TRACE_SYSTEM_PROMPTS = {
    task_name: _system_prompt_for(task_name) for task_name in TASK_TRACE_CONTRACTS
}


def get_task_trace_contract(task_name: str) -> Dict[str, Any]:

    base = TASK_TRACE_CONTRACTS.get(task_name)
    if base is None:
        return {
            "family": "unknown",
            "trace_goal": "Produce a grounded causal reasoning trace.",
            "required_moves": [
                "Ground observations in objects, positions, and states.",
                "Trace causal mechanisms from preconditions to effects.",
                "Disambiguate a plausible alternative.",
            ],
            "min_words": 200,
            "validation_clusters": deepcopy(_BASE_CLUSTERS),
        }
    return deepcopy(base)


def get_system_prompt(task_name: str) -> str:

    return CAUSAL_TRACE_SYSTEM_PROMPTS.get(task_name, _COMMON_PREAMBLE)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _plan_context_lines(plan_context: dict | None) -> List[str]:
    if not plan_context:
        return []

    lines: List[str] = []
    if plan_context.get("high_level_goal"):
        lines.append(f"high_level_goal: {plan_context['high_level_goal']}")

    for key in ["previous_step", "step", "next_step"]:
        step = plan_context.get(key)
        if not isinstance(step, dict):
            continue
        label = key
        lines.append(f"{label}.step_id: {step.get('step_id', '?')}")
        lines.append(f"{label}.step_goal: {step.get('step_goal', '?')}")
        if step.get("rationale"):
            lines.append(f"{label}.rationale: {step.get('rationale')}")
        causal_chain = step.get("causal_chain")
        if isinstance(causal_chain, dict):
            for field in [
                "agent",
                "action",
                "patient",
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ]:
                value = causal_chain.get(field)
                if value:
                    lines.append(f"{label}.causal_chain.{field}: {_format_value(value)}")

    all_step_goals = plan_context.get("all_step_goals")
    if isinstance(all_step_goals, list) and all_step_goals:
        compact = " | ".join(str(goal) for goal in all_step_goals[:20])
        lines.append(f"ordered_plan_step_goals: {compact}")

    return lines


def build_causal_trace_user_prompt(
    *,
    task_name: str,
    question: str,
    answer: str,
    llm_fields: dict,
    plan_context: dict | None = None,
) -> str:

    contract = get_task_trace_contract(task_name)
    family = _FAMILY_GUIDANCE.get(contract["family"], {})

    answer_words = len(answer.split())
    if answer_words <= 30:
        depth_note = (
            f"The target answer is short ({answer_words} words), so the trace must be substantially deeper than the answer. "
            "Expand the hidden perceptual chain, physical mechanism, and disambiguation."
        )
    elif answer_words <= 80:
        depth_note = (
            f"The target answer is medium length ({answer_words} words). The trace must explain the causal basis for each main claim."
        )
    else:
        depth_note = (
            f"The target answer is detailed ({answer_words} words). The trace should audit the most important causal links without copying the answer wording."
        )

    parts: List[str] = [
        "=== TASK CAUSAL TRACE CONTRACT ===",
        f"Task: {task_name}",
        f"Family: {contract['family']}",
        f"Family meaning: {family.get('description', 'grounded causal reasoning')}",
        f"Trace goal: {contract['trace_goal']}",
        "",
        "Required reasoning moves for this task:",
        *(f"- {move}" for move in contract["required_moves"]),
        "",
        "Family-level reasoning moves:",
        *(f"- {move}" for move in family.get("global_moves", [])),
        "",
        f"Minimum target depth: {contract['min_words']} words.",
        depth_note,
        "",
        "=== QUESTION ===",
        question,
        "",
        "=== STRUCTURED VISUAL AND CAUSAL EVIDENCE ===",
    ]

    if llm_fields:
        for key, value in llm_fields.items():
            parts.append(f"{key}: {_format_value(value)}")
    else:
        parts.append("(No extra structured fields were provided; rely on the question, answer, and plan context.)")

    plan_lines = _plan_context_lines(plan_context)
    if plan_lines:
        parts.extend(["", "=== PLAN CONTEXT ===", *plan_lines])

    parts.extend([
        "",
        "=== TARGET ANSWER KEPT OUTSIDE THE TRACE ===",
        answer,
        "",
        "Write only the causal reasoning trace. Start with concrete visual/state evidence, build the causal chain, include the task-specific moves above, and stop on the decisive causal mechanism. Do not output <think> tags. Do not copy the answer verbatim.",
    ])
    return "\n".join(parts)



def build_cot_user_prompt(
    *,
    task_name: str,
    question: str,
    answer: str,
    llm_fields: dict,
    plan_context: dict | None = None,
) -> str:
    return build_causal_trace_user_prompt(
        task_name=task_name,
        question=question,
        answer=answer,
        llm_fields=llm_fields,
        plan_context=plan_context,
    )


ALL_TASK_NAMES = list(TASK_NAMES)
