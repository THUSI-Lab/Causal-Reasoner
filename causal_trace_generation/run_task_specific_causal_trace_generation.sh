#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-data.jsonl}"
PARALLEL_API="${PARALLEL_API:-4}"
TASKS="${TASKS:-}"
EVIDENCE_MODE="${EVIDENCE_MODE:-auto}"
MAX_IMAGES="${MAX_IMAGES:-4}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-3}"
VISUAL_IO_PARALLEL="${VISUAL_IO_PARALLEL:-4}"
MODEL="${MODEL:-gpt-5.4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
TIMEOUT="${TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-3}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "Set SOURCE_ROOT to a QA directory containing Task_*/data.jsonl files." >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "Set OUTPUT_ROOT to the causal-trace output directory." >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  find "${SOURCE_ROOT}" -path '*/Task_*/data.jsonl' -type f -print
  exit 0
fi

ARGS=(
  "${SCRIPT_DIR}/generate_task_specific_causal_traces.py"
  --input-dir "${SOURCE_ROOT}"
  --output-dir "${OUTPUT_ROOT}"
  --parallel-api "${PARALLEL_API}"
  --resume
  --keep-going
  --model "${MODEL}"
  --reasoning-effort "${REASONING_EFFORT}"
  --max-retries "${MAX_RETRIES}"
  --timeout "${TIMEOUT}"
  --evidence-mode "${EVIDENCE_MODE}"
  --max-images "${MAX_IMAGES}"
  --max-video-frames "${MAX_VIDEO_FRAMES}"
  --visual-io-parallel "${VISUAL_IO_PARALLEL}"
  --output-filename "${OUTPUT_FILENAME}"
)

if [[ -n "${TASKS}" ]]; then
  ARGS+=(--tasks "${TASKS}")
fi

python3 "${ARGS[@]}"
