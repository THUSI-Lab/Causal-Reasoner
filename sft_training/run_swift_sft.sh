#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${SFT_MODEL:?Set SFT_MODEL to a model name or checkpoint path.}"
: "${SFT_DATASET:?Set SFT_DATASET to a SWIFT-format JSONL dataset path.}"

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-../sft_outputs/swift_sft}"
SFT_TUNER_TYPE="${SFT_TUNER_TYPE:-lora}"
SFT_TORCH_DTYPE="${SFT_TORCH_DTYPE:-bfloat16}"
SFT_SPLIT_DATASET_RATIO="${SFT_SPLIT_DATASET_RATIO:-0.01}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-2}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-4}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-100}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-5}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"
SFT_TRUNCATION_STRATEGY="${SFT_TRUNCATION_STRATEGY:-left}"
SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"
SFT_DATALOADER_NUM_WORKERS="${SFT_DATALOADER_NUM_WORKERS:-4}"
SFT_DATASET_NUM_PROC="${SFT_DATASET_NUM_PROC:-8}"
SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-true}"
SFT_SAVE_ONLY_MODEL="${SFT_SAVE_ONLY_MODEL:-false}"
SFT_REPORT_TO="${SFT_REPORT_TO:-tensorboard}"
SFT_LOAD_FROM_CACHE_FILE="${SFT_LOAD_FROM_CACHE_FILE:-true}"

ARGS=(
  sft
  --model "$SFT_MODEL"
  --dataset "$SFT_DATASET"
  --output_dir "$SFT_OUTPUT_DIR"
  --tuner_type "$SFT_TUNER_TYPE"
  --torch_dtype "$SFT_TORCH_DTYPE"
  --split_dataset_ratio "$SFT_SPLIT_DATASET_RATIO"
  --num_train_epochs "$SFT_NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$SFT_PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$SFT_PER_DEVICE_EVAL_BATCH_SIZE"
  --learning_rate "$SFT_LEARNING_RATE"
  --gradient_accumulation_steps "$SFT_GRADIENT_ACCUMULATION_STEPS"
  --eval_steps "$SFT_EVAL_STEPS"
  --save_steps "$SFT_SAVE_STEPS"
  --save_total_limit "$SFT_SAVE_TOTAL_LIMIT"
  --logging_steps "$SFT_LOGGING_STEPS"
  --max_length "$SFT_MAX_LENGTH"
  --truncation_strategy "$SFT_TRUNCATION_STRATEGY"
  --warmup_ratio "$SFT_WARMUP_RATIO"
  --dataloader_num_workers "$SFT_DATALOADER_NUM_WORKERS"
  --dataset_num_proc "$SFT_DATASET_NUM_PROC"
  --gradient_checkpointing "$SFT_GRADIENT_CHECKPOINTING"
  --save_only_model "$SFT_SAVE_ONLY_MODEL"
  --report_to "$SFT_REPORT_TO"
  --load_from_cache_file "$SFT_LOAD_FROM_CACHE_FILE"
)

if [[ -n "${SFT_TEMPLATE:-}" ]]; then
  ARGS+=(--template "$SFT_TEMPLATE")
fi

if [[ -n "${SFT_AGENT_TEMPLATE:-}" ]]; then
  ARGS+=(--agent_template "$SFT_AGENT_TEMPLATE")
fi

if [[ -n "${SFT_LOSS_SCALE:-}" ]]; then
  ARGS+=(--loss_scale "$SFT_LOSS_SCALE")
fi

if [[ "$SFT_TUNER_TYPE" != "full" ]]; then
  SFT_LORA_RANK="${SFT_LORA_RANK:-16}"
  SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}"
  SFT_TARGET_MODULES="${SFT_TARGET_MODULES:-all-linear}"
  if [[ -n "$SFT_LORA_RANK" ]]; then
    ARGS+=(--lora_rank "$SFT_LORA_RANK")
  fi
  if [[ -n "$SFT_LORA_ALPHA" ]]; then
    ARGS+=(--lora_alpha "$SFT_LORA_ALPHA")
  fi
  if [[ -n "$SFT_TARGET_MODULES" ]]; then
    ARGS+=(--target_modules "$SFT_TARGET_MODULES")
  fi
fi

if [[ -n "${SFT_DEEPSPEED:-}" ]]; then
  ARGS+=(--deepspeed "$SFT_DEEPSPEED")
fi

if [[ -n "${SFT_ATTN_IMPL:-}" ]]; then
  ARGS+=(--attn_impl "$SFT_ATTN_IMPL")
fi

if [[ -n "${SFT_FREEZE_VIT:-}" ]]; then
  ARGS+=(--freeze_vit "$SFT_FREEZE_VIT")
fi

if [[ -n "${SFT_FREEZE_ALIGNER:-}" ]]; then
  ARGS+=(--freeze_aligner "$SFT_FREEZE_ALIGNER")
fi

if [[ -n "${SFT_DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
  ARGS+=(--ddp_find_unused_parameters "$SFT_DDP_FIND_UNUSED_PARAMETERS")
fi

swift "${ARGS[@]}" "$@"
