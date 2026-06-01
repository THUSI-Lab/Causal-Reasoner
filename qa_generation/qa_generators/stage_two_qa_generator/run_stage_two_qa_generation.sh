#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/qa_output}"
PARALLEL_API="${PARALLEL_API:-2}"
LIMIT="${LIMIT:-0}"
MODEL_MODE="${MODEL_MODE:-api}"

if [[ -z "${INPUT_ROOT}" ]]; then
  echo "Set INPUT_ROOT to a directory containing stage two final_plan.json item folders." >&2
  exit 2
fi

ARGS=(
  "${SCRIPT_DIR}/generate_stage_two_qa.py"
  --input-root "${INPUT_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --parallel-api "${PARALLEL_API}"
  --keep-going
  --resume
)

if [[ "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

case "${MODEL_MODE}" in
  api)
    ARGS+=(--require-llm)
    ;;
  draft)
    ARGS+=(--no-api)
    ;;
  *)
    echo "MODEL_MODE must be either 'api' or 'draft'." >&2
    exit 2
    ;;
esac

python3 "${ARGS[@]}"
