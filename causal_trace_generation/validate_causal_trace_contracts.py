


from __future__ import annotations

from causal_trace_prompts import (
    ALL_TASK_NAMES,
    build_causal_trace_user_prompt,
    get_system_prompt,
    get_task_trace_contract,
)
from generate_task_specific_causal_traces import _check_causal_trace_contract


def main() -> None:
    assert len(ALL_TASK_NAMES) == 20, f"expected 20 tasks, got {len(ALL_TASK_NAMES)}"
    for task_name in ALL_TASK_NAMES:
        contract = get_task_trace_contract(task_name)
        assert contract["family"], task_name
        assert contract["trace_goal"], task_name
        assert len(contract["required_moves"]) >= 3, task_name
        assert contract["min_words"] >= 200, task_name
        assert contract["validation_clusters"], task_name
        assert "causal reasoning traces" in get_system_prompt(task_name), task_name

        prompt = build_causal_trace_user_prompt(
            task_name=task_name,
            question="What is happening and why?",
            answer="The action succeeds because the required conditions are satisfied.",
            llm_fields={
                "step_goal": "Move the object into the reachable target area.",
                "mechanism": "The hand applies force through contact with the object surface.",
            },
            plan_context={
                "high_level_goal": "Complete the activity.",
                "all_step_goals": ["Prepare the object.", "Move the object.", "Finish the activity."],
            },
        )
        assert "TASK CAUSAL TRACE CONTRACT" in prompt, task_name
        assert "TARGET ANSWER KEPT OUTSIDE THE TRACE" in prompt, task_name

    sample_trace = (
        "The visible object is positioned within reach, and its state makes the next motion possible. "
        "Because the hand can maintain contact with the rigid surface, the applied force transfers through the object instead of slipping away. "
        "That contact path enables the action mechanism: force changes the object's spatial position, which then creates the needed postcondition for the next step. "
        "If the object were outside the reachable area or if the surface were too unstable, the same motion would fail because the agent could not preserve contact. "
        "This alternative is ruled out by the observed position, state, and contact relationship, so the causal dependency runs from precondition to mechanism to effect."
    )
    ok, reason = _check_causal_trace_contract(sample_trace, "Task_05_Holistic_Causal_Chain")
    assert ok, reason
    print("causal trace contract checks passed")


if __name__ == "__main__":
    main()
