# QA Generation

This directory contains the active QA generation code and the 20-task prompt registry.

## Layout

```text
prompts_and_task_specs/
  qa_task_prompt_registry.py

qa_generators/
  stage_one_qa_generator/
  stage_two_qa_generator/
```

## QA Filtering

QA filtering is maintained separately in `../qa_filtering/`.

## Task Coverage

The prompt registry covers 20 task families spanning spatial preconditions, affordance preconditions, physical feasibility, affordance visual semantics, postconditions, state evolution, strategic rationale, inter-step dependency, bad-plan diagnosis and repair, counterfactual outcome, and failure recovery.

## Data Policy

This directory stores code only. It does not include generated QA data, raw media, model outputs, runtime caches, or local execution artifacts.

All prompt and task content used by the retained QA code is embedded in Python source files.

## Dependencies

Azure-backed QA generation requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_PROFILE` or `API_PROFILE`.
