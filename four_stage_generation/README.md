# Four-Stage Generation Pipelines

This directory contains the single-step and multi-step four-stage data generation pipelines used to convert raw visual episodes into structured plans, localized steps, refined keyframes, and atomic action annotations.

## Layout

```text
single_step/
  generate_single_step_four_stage_dataset.py
  stage1_plan_draft_generator.py
  stage2_step_localizer.py
  stage3_refine_keyframes.py
  stage4_atomic_action_generator.py
  validate_four_stage_output.py
  single_step_prompt_templates.py
  four_stage_prompt_templates.py
  four_stage_common.py
  azure_openai_client.py

multi_step/
  run_multi_step_four_stage_pipeline.py
  stage1_plan_draft_generator.py
  stage2_step_localizer.py
  stage3_refine_keyframes.py
  stage4_atomic_action_generator.py
  validate_four_stage_output.py
  four_stage_prompt_templates.py
  four_stage_common.py
  azure_openai_client.py
```

## Entry Points

Use `single_step/generate_single_step_four_stage_dataset.py` for single-step videos.

Use `multi_step/run_multi_step_four_stage_pipeline.py` for multi-step generation.

## Pipeline Contract

The retained pipeline follows four stages:

1. Draft a high-level causal plan from visual evidence.
2. Localize plan steps to visual frame or clip spans.
3. Refine keyframes and visual anchors for each step.
4. Generate atomic action records with preconditions, effects, and validation metadata.

The validators check schema consistency, step ordering, evidence availability, keyframe references, and causal fields needed by downstream QA generation.

## Data Policy

No generated datasets, raw videos, model outputs, local caches, or machine-specific launch scripts are stored in this directory.

## Dependencies

Azure-backed generation requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_PROFILE` or `API_PROFILE`.
