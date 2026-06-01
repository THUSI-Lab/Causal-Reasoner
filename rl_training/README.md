# Causal Learner Reward

This directory contains the standalone Causal Learner reward implementation used by the RL stage.

No training data, model checkpoint, runtime cache, output checkpoint, machine path, cluster submission script, or environment-specific activation command is stored here.

## Files

- `causal_learner_reward/compute_score.py`: reward entry points compatible with RL training runners.
- `causal_learner_reward/data_schema.py`: conversion and validation helpers for reward-model ground truth rows.
- `causal_learner_reward/rule_reward.py`: deterministic rule reward for structured task-specific reference fields.
- `causal_learner_reward/judge_prompts.py`: task-specific strict multimodal judge prompts.
- `causal_learner_reward/rubric_scoring.py`: parser and scoring logic for judge JSON output.
- `causal_learner_reward/judge_client.py`: OpenAI-compatible multimodal judge client with retry, timeout, cache, and endpoint-pool support.
- `causal_learner_reward/video_payload.py`: video or sampled-frame payload preparation for judge calls.
- `causal_learner_reward/config.py`: reward and judge configuration dataclasses.

## Supported Tasks

The released reward covers the QA tasks used by the RL stage:

```text
Task_01 Spatial Precondition
Task_02 Affordance Precondition
Task_06 Spatial Postcondition
Task_07 Affordance Postcondition
Task_18 Bad Plan Diagnosis And Repair
Task_19 Counterfactual Outcome
Task_20 Failure Recovery
```

## Reward Logic

Phase 1 computes a deterministic task-specific rule reward from structured reference fields.

Phase 2 keeps the rule reward and adds a strict multimodal rubric judge. The judge evaluates the model output against the question, ground truth reference fields, and visual evidence, then returns JSON-only rubric decisions. The final score combines the rule and judge components using the task-specific alpha table in `config.py`.

Judge failures, invalid judge JSON, missing visual evidence, unsupported task ids, or malformed ground truth fail closed with the configured failure score.

## Minimal Use

```python
from causal_learner_reward import compute_score

result = compute_score(
    data_source="causal_learner",
    solution_str="<model output>",
    ground_truth="<ground truth JSON string>",
    extra_info={
        "task_id": "Task_20",
        "question": "<question text>",
        "video_path": "<video path>",
    },
    phase=1,
)
```

For judge-backed scoring, pass `phase=2` and a `judge_api_config` mapping:

```python
result = compute_score(
    data_source="causal_learner",
    solution_str="<model output>",
    ground_truth="<ground truth JSON string>",
    extra_info={
        "task_id": "Task_20",
        "question": "<question text>",
        "video_path": "<video path>",
    },
    phase=2,
    judge_api_config={
        "mode": "vllm",
        "base_url": "<OPENAI_COMPATIBLE_ENDPOINT>",
        "model_name": "<JUDGE_MODEL>",
        "video_transport": "native_video",
    },
)
```

## Install

```bash
python -m pip install -r requirements.txt
```

The rule-only path uses the Python standard library. External judge calls require the dependencies listed in `requirements.txt`. Frame-fallback video preprocessing expects the surrounding RL stack to provide the video utility module used by the training runner.
