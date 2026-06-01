# Standalone Evaluation

This directory contains benchmark evaluation code without benchmark data.

## Files

- `evaluate_mcq_benchmark.py`: evaluates MCQ items.
- `evaluate_open_qa_with_rubric_judge.py`: generates open-QA answers and scores them with task-specific rubric prompts.
- `evaluation_common.py`: shared loading, media handling, Azure OpenAI calling, prompt validation, and output helpers.
- `open_qa_judge_rubric_prompts_en.py`: task-specific open-QA rubric judge prompts.
- `open_qa_model_registry.json`: model registry.
- `validate_benchmark_prompts_and_data.sh`: validates data, media references, prompts, and dry-run evaluation logic.
- `run_full_benchmark_evaluation.sh`: runs the full MCQ and open-QA evaluation.

## Data Location

No benchmark data is stored in this directory.

By default, scripts look for data at:

```text
../benchmark_data
```

For any other data location, set:

```bash
BENCHMARK_DATA_ROOT=<BENCHMARK_DATA_DIR>
```

The data directory is expected to contain `mcq/`, `qa/`, and `multimodal_data/`.

## Validate

```bash
cd evaluation
bash validate_benchmark_prompts_and_data.sh
```

## Run Full Evaluation

```bash
cd evaluation
bash run_full_benchmark_evaluation.sh
```

For a different output location:

```bash
EVALUATION_OUTPUT_ROOT=<OUTPUT_DIR> bash run_full_benchmark_evaluation.sh
```

## Dependencies

```bash
python -m pip install -r requirements.txt
command -v ffmpeg
command -v ffprobe
```

Real model-backed evaluation requires Azure OpenAI credentials plus `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_PROFILE` or `API_PROFILE`.
