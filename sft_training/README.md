# SFT Training

This directory contains the standalone supervised fine-tuning entry points used with ModelScope SWIFT.

No training data, model checkpoint, runtime cache, output checkpoint, machine path, cluster submission script, or environment-specific activation command is stored here.

## Files

- `prepare_swift_sft_dataset.py`: converts QA-style JSONL rows into the SWIFT `messages` JSONL format.
- `run_swift_sft.sh`: launches `swift sft` with environment-variable configuration.
- `requirements.txt`: minimal dependency entry for SWIFT installation.

## Data Format

The prepared dataset follows the SWIFT custom SFT format:

```json
{"messages":[{"role":"user","content":"<question>"},{"role":"assistant","content":"<answer>"}]}
```

For multimodal rows, the converter preserves `images` and `videos` fields and inserts missing `<image>` or `<video>` tags into the user message.

## Prepare Data

```bash
python prepare_swift_sft_dataset.py \
  --input-jsonl <QA_JSONL> \
  --output-jsonl <SWIFT_SFT_JSONL>
```

For media paths relative to a data root:

```bash
python prepare_swift_sft_dataset.py \
  --input-jsonl <QA_JSONL> \
  --output-jsonl <SWIFT_SFT_JSONL> \
  --media-root <MEDIA_ROOT> \
  --path-mode absolute
```

## Run SFT

```bash
python -m pip install -r requirements.txt
SFT_MODEL=<MODEL_OR_CHECKPOINT> \
SFT_DATASET=<SWIFT_SFT_JSONL> \
bash run_swift_sft.sh
```

Common optional settings:

```bash
SFT_TEMPLATE=qwen3 \
SFT_TUNER_TYPE=lora \
SFT_OUTPUT_DIR=<OUTPUT_DIR> \
SFT_NUM_TRAIN_EPOCHS=2 \
SFT_PER_DEVICE_TRAIN_BATCH_SIZE=1 \
SFT_GRADIENT_ACCUMULATION_STEPS=8 \
bash run_swift_sft.sh
```

Additional SWIFT arguments can be appended after the script command.
