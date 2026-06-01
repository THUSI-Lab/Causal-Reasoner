set -euo pipefail

cd "$(dirname "$0")"
REPOSITORY_ROOT="$(cd .. && pwd)"
BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-$REPOSITORY_ROOT/benchmark_data}"
OUT_ROOT="${EVALUATION_OUTPUT_ROOT:-$REPOSITORY_ROOT/eval_outputs/full_benchmark}"

python evaluate_mcq_benchmark.py \
  --benchmark-data-root "$BENCHMARK_DATA_ROOT" \
  --model gpt-5.4 \
  --max-concurrency 4 \
  --output-dir "$OUT_ROOT/mcq_eval"

python evaluate_open_qa_with_rubric_judge.py \
  --benchmark-data-root "$BENCHMARK_DATA_ROOT" \
  --model gpt-5.4 \
  --judge-model gpt-5.4 \
  --max-concurrency 4 \
  --output-dir "$OUT_ROOT/qa_rubric_judge"
