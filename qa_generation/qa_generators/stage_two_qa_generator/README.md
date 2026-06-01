# Stage-Two QA Generator

This directory contains the stage-two QA generation code for item directories that contain `final_plan.json`.

## Files

- `generate_stage_two_qa.py`: main stage-two QA generator.
- `generate_stage_two_open_qa.py`: open-QA generation helper.
- `run_stage_two_qa_generation.sh`: shell entry point for batch generation.
- `backfill_qa_evidence_paths_indexed.py`: evidence-path reconstruction using indexed metadata.
- `backfill_qa_evidence_paths_rule_based.py`: evidence-path reconstruction using rule-based matching.
- `backfill_qa_evidence_paths_verified.py`: verified evidence-path backfill workflow.
- `normalize_final_plan_identifier_tokens.py`: normalizes final-plan identifier tokens.
- `prune_non_plan_item_dirs.py`: removes item directories that do not contain valid plan inputs.
- `azure_openai_client.py`: Azure OpenAI client helper.

## Usage

```bash
INPUT_ROOT=<FINAL_PLAN_ITEM_DIR> OUTPUT_ROOT=<QA_OUTPUT_DIR> \
  bash run_stage_two_qa_generation.sh
```

## Output

The generator writes task-organized JSONL outputs under `Task_*/data.jsonl`. Evidence backfill scripts preserve package-relative media references when the input metadata is available.

No generated QA data, raw media, runtime cache, or machine-specific launch file is stored in this directory.
