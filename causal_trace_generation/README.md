# Causal Trace Generation

This directory contains task-specific utilities for adding causal reasoning traces to generated QA rows. The traces are designed to explain the visual state, causal dependency, physical feasibility, and task-specific reasoning needed to support each answer.

## Files

- `generate_task_specific_causal_traces.py`: main trace generation entry point.
- `causal_trace_prompts.py`: task-specific trace contracts and prompt templates.
- `causal_trace_task_prompt_registry.py`: compact task prompt registry.
- `causal_trace_prompt_adapter.py`: adapters for converting QA rows into trace prompts.
- `reviewer_causal_trace_rubric_prompts.py`: task-specific reviewer rubrics for trace quality checks.
- `validate_causal_trace_contracts.py`: local validation checks for trace schema and task contracts.
- `run_task_specific_causal_trace_generation.sh`: shell entry point for batch generation.
- `azure_openai_client.py`: Azure OpenAI client helper.

## Input And Output

The expected input layout is a QA directory containing `Task_*/data.jsonl` files. Each row should contain a question, answer, task identifier, visual evidence reference, and source context fields when available.

The output mirrors the task directory layout and writes JSONL rows augmented with trace metadata.

## Usage

```bash
SOURCE_ROOT=<QA_INPUT_DIR> OUTPUT_ROOT=<QA_WITH_TRACES_DIR> \
  bash causal_trace_generation/run_task_specific_causal_trace_generation.sh
```

## Quality Checks

The generation code applies task-specific trace contracts and rejects traces that are empty, generic, visually ungrounded, missing required causal elements, or inconsistent with the target task family.

Run the local contract check with:

```bash
python causal_trace_generation/validate_causal_trace_contracts.py
```

## Dependencies

Azure-backed causal trace generation requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_PROFILE` or `API_PROFILE`.
