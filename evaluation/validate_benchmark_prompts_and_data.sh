set -euo pipefail

cd "$(dirname "$0")"
REPOSITORY_ROOT="$(cd .. && pwd)"
BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-$REPOSITORY_ROOT/benchmark_data}"
OUT_ROOT="${EVALUATION_OUTPUT_ROOT:-$REPOSITORY_ROOT/eval_outputs/validation_dryrun}"
VALIDATION_DRY_RUN_LIMIT_PER_TASK="${VALIDATION_DRY_RUN_LIMIT_PER_TASK:-1}"
export BENCHMARK_DATA_ROOT

python -m py_compile evaluation_common.py evaluate_mcq_benchmark.py evaluate_open_qa_with_rubric_judge.py open_qa_judge_rubric_prompts_en.py

python - <<'PY'
import os
from pathlib import Path
from collections import Counter
import evaluation_common as r
import open_qa_judge_rubric_prompts_en as prompts

benchmark_data_root = Path(os.environ["BENCHMARK_DATA_ROOT"])
prompt_check = r.validate_prompt_module()
print("prompt_module_check_passed", prompt_check["module_check_passed"])
print("prompt_count", prompt_check["prompt_count"])

for split in ["mcq", "qa"]:
    items = r.load_items(benchmark_data_root, split, None)
    by_task = Counter(item.task_name for item in items)
    missing_media = []
    prompt_fail = []
    for item in items:
        image_paths, video_paths = r.collect_existing_media_paths(item)
        if not image_paths and not video_paths:
            missing_media.append((item.task_name, item.sample_id))
        if split == "qa":
            prompt_task_name = (item.raw.get("meta") or {}).get("task_name") or item.task_name
            try:
                prompts.get_prompt(prompt_task_name)
            except Exception as exc:
                prompt_fail.append((item.task_name, item.sample_id, str(exc)))
    print(split, "total", len(items), "by_task", dict(sorted(by_task.items())))
    print(split, "missing_media", len(missing_media), "prompt_fail", len(prompt_fail))
    if missing_media or prompt_fail:
        raise SystemExit(f"{split} validation failed")
PY

python evaluate_mcq_benchmark.py \
  --benchmark-data-root "$BENCHMARK_DATA_ROOT" \
  --dry-run \
  --limit-per-task "$VALIDATION_DRY_RUN_LIMIT_PER_TASK" \
  --video-max-frames 2 \
  --video-clip-frames 1 \
  --max-concurrency 1 \
  --output-dir "$OUT_ROOT/mcq_eval" \
  --overwrite

python evaluate_open_qa_with_rubric_judge.py \
  --benchmark-data-root "$BENCHMARK_DATA_ROOT" \
  --dry-run \
  --limit-per-task "$VALIDATION_DRY_RUN_LIMIT_PER_TASK" \
  --video-max-frames 2 \
  --video-clip-frames 1 \
  --max-concurrency 1 \
  --output-dir "$OUT_ROOT/qa_rubric_judge" \
  --overwrite
