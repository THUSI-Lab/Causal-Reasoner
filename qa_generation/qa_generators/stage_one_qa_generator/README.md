# Stage One QA Generator

This package generates QA for stage one, usually single-step, plan outputs stored as `causal_plan_with_keyframes.json`.

## Key Files

- `generate_stage_one_qa.py` contains the shared generator logic for this stage one task set.
- `generate_stage_one_qa_parallel.py` is the recommended parallel entry point. It distributes item processing, supports resume mode, and writes audit/issue reports.
- `azure_openai_client.py` configures Azure OpenAI access through Azure CLI credentials.

## Typical Command

```bash
python3 generate_stage_one_qa_parallel.py \
  --input-root <STAGE_ONE_PLAN_DIR> \
  --output-dir <QA_OUTPUT_DIR> \
  --workers 8 \
  --keep-going \
  --resume \
  --text-only \
  --min-steps 1 \
  --require-llm \
  --llm-require-success \
  --llm-temperature 0.0 \
  --llm-verify warn
```

Use `--limit` for smoke tests, `--no-api` for deterministic drafts, and `--audit-only` to validate an existing output directory.

## Outputs and Checks

Outputs are written under `<output-dir>/Task_*/data.jsonl`, with `issues_*`, `audit_report.json`, `deep_audit_report.json`, and resume state files.

No generated QA data, media files, runtime cache, or machine-specific launch file is stored in this directory.
