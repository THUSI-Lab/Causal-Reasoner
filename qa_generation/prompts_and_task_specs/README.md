# Prompt And Task Specifications

This directory contains the active 20-task QA prompt registry used by the QA generation code.

## File

- `qa_task_prompt_registry.py`: canonical task keys, public task names, evidence types, question intents, and prompt text used by the retained generators.

## Contract

The registry is the source of truth for task-specific QA generation prompts. Filtering logic is intentionally kept outside this directory under `qa_filtering/`.
