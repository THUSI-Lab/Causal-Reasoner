<div align="center">

# *Token Predictors Are Not Planners*: Building Physically Grounded Causal Reasoners

Zheng Lu<sup>1,2,*</sup>, Mingqi Gao<sup>1,*</sup>, Qinlei Xie<sup>1,*</sup>, Wanqi Zhong<sup>1</sup>, Hanwen Cui<sup>1</sup>, Heng Cao<sup>1</sup>, Zirui Song<sup>3</sup>, Yifan Yang<sup>2</sup>, Chong Luo<sup>2</sup>, Bei Liu<sup>2,&dagger;</sup>, Yiming Li<sup>1,&dagger;</sup>

<sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>2</sup>Microsoft Research Asia &nbsp;&nbsp; <sup>3</sup>MBZUAI

<sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding authors

<p>
  <a href="https://huggingface.co/Lululzz/Causal_Planner">
    <img src="https://img.shields.io/badge/Model-Causal_Planner-yellow?style=flat-square" alt="Model">
  </a>
  <a href="https://huggingface.co/datasets/anonymous-causal-plan/Causal_Plan/tree/main/CausalPlan-1M-QA">
    <img src="https://img.shields.io/badge/Dataset-CausalPlan--1M--QA-blue?style=flat-square" alt="Dataset">
  </a>
  <a href="https://huggingface.co/datasets/anonymous-causal-plan/Causal_Plan/tree/main/Causal-Plan-Bench">
    <img src="https://img.shields.io/badge/Benchmark-Causal--Plan--Bench-orange?style=flat-square" alt="Benchmark">
  </a>
</p>

[Overview](#overview) •
[Evaluation](#evaluation) •
[Generation And Filtering](#generation-and-filtering) •
[Training Utilities](#training-utilities) •
[Repository Layout](#repository-layout)

<img src="assets/paradigm_comparison.jpg" alt="Paradigm comparison for causal planning" width="96%">

</div>

## Overview

This repository is the code release for **Token Predictors Are Not Planners: Building Physically Grounded Causal Reasoners**. The project studies embodied planning as physically grounded causal reasoning rather than surface-level next-token continuation, focusing on the observation that plausible action sequences can still fail when they ignore hidden preconditions, object affordances, state transitions, temporal dependencies, or recovery constraints.

In the paper, Causal Plan denotes the framework and released resources. The code supports the main technical components described in the accompanying paper:

- `Causal-Plan-Bench`: a 1,200-instance diagnostic suite across 12 benchmark tasks for physically grounded planning.
- `Causal-Plan-1M`: a million-scale causal supervision pipeline spanning 20 task families with task-specific reasoning traces.
- `Causal Planner`: a training setup that combines staged SFT preparation with task-specific RL reward scoring.
- QA filtering and auditing code for visual grounding, logical coherence, and physical feasibility.

Benchmark data, training data, raw videos, model checkpoints, generated outputs, runtime caches, and machine-specific job scripts are intentionally not stored in this repository.

## Causal Dimensions

The paper decomposes embodied planning into four diagnostic dimensions:

| Dimension | What It Checks |
| --- | --- |
| Executability | Whether the current visual state satisfies spatial and affordance preconditions before an action is attempted. |
| Effects | Whether the model understands the state changes caused by an action. |
| Composition | Whether local steps form a coherent long-horizon causal order. |
| Robustness | Whether the model can diagnose failures, reason counterfactually, and recover from disrupted plans. |

These dimensions are reflected throughout the evaluation prompts, QA generation templates, causal trace contracts, and reward rubrics in this repository.

<p align="center">
  <img src="assets/data_generation_pipeline.png" alt="Data generation and curation pipeline" width="96%">
</p>

## What Is Released

| Paper Component | Released Code | External Artifacts |
| --- | --- | --- |
| Causal-Plan-Bench | MCQ evaluation, open-QA generation, task-specific rubric judging, data validation, and model registry utilities for the 12-task benchmark protocol. | Benchmark data and media are not bundled. |
| Causal-Plan-1M | Four-stage generation, 20-task QA generation, QA filtering, and causal trace generation code. | Generated training data and raw source videos are not bundled. |
| Causal Planner SFT | SWIFT-compatible data conversion and configurable SFT launch wrapper. | Training data, checkpoints, and machine-specific launch scripts are not bundled. |
| Causal Planner RL | Standalone Causal Learner reward package with rule rewards and optional multimodal rubric judging. | Rollout data, RL runner integration, and checkpoints are not bundled. |

## External Data Layout

Evaluation expects an external benchmark data directory:

```text
benchmark_data/
  mcq/
  qa/
  multimodal_data/
```

Place this directory at `benchmark_data/` under the repository root, or set `BENCHMARK_DATA_ROOT` explicitly.

## Getting Started

Clone the repository and install dependencies only for the module you need. For benchmark evaluation:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIR>/evaluation
python -m pip install -r requirements.txt
```

Model-backed generation, filtering, trace generation, and open-QA evaluation require the relevant Azure OpenAI or OpenAI-compatible endpoint credentials documented in each module README.

## Evaluation

Causal-Plan-Bench is described in the paper as a held-out 1,200-instance diagnostic suite across 12 benchmark tasks. It uses a dual-format evaluation protocol: deterministic executability and effects tasks use MCQ scoring, while composition and robustness tasks use open-ended answers judged by task-specific causal rubrics.

Validate prompt definitions, data structure, media references, and dry-run evaluation logic:

```bash
cd evaluation
BENCHMARK_DATA_ROOT=<BENCHMARK_DATA_DIR> \
bash validate_benchmark_prompts_and_data.sh
```

Run the full benchmark evaluation:

```bash
cd evaluation
BENCHMARK_DATA_ROOT=<BENCHMARK_DATA_DIR> \
bash run_full_benchmark_evaluation.sh
```

The validation script compiles the evaluation code, checks task-specific open-QA judge prompts, verifies MCQ and open-QA rows, checks media availability, and runs dry-run evaluation without model API calls. The full evaluation script runs MCQ evaluation and open-QA generation plus rubric judging. Model aliases and provider settings are defined in `evaluation/open_qa_model_registry.json`.

<p align="center">
  <img src="assets/performance_overview.png" alt="Benchmark performance overview" width="96%">
</p>

### Main In-Domain Results

The overall score is the macro-average across the 12 Causal-Plan-Bench tasks. Bold marks the best result and underline marks the second-best result where applicable.

<p align="center">
  <img src="assets/table_main_indomain_results.png" alt="Main in-domain results on Causal-Plan-Bench" width="96%">
</p>

### Cross-Benchmark Transfer

The average is the arithmetic mean over EgoPlan-Bench2, RoboVQA, and Cosmos-Reason.

<p align="center">
  <img src="assets/table_cross_benchmark_transfer.png" alt="Cross-benchmark transfer results" width="96%">
</p>

### Ablation Results

Scores follow the same four-dimension Causal-Plan-Bench protocol.

<p align="center">
  <img src="assets/table_ablation_results.png" alt="Causal Planner ablation results" width="96%">
</p>

## Generation And Filtering

The data construction code follows the paper's four-stage protocol:

| Stage | Released Entry Points |
| --- | --- |
| Global blueprinting | `stage1_plan_draft_generator.py` |
| Temporal grounding | `stage2_step_localizer.py` |
| Causal enrichment | `stage3_refine_keyframes.py` |
| Atomic decomposition | `stage4_atomic_action_generator.py` |

The QA generation and causal trace code covers the 20 task families used for Causal-Plan-1M-style supervision, with task-specific visual evidence formats, answer constraints, and trace contracts.

Four-stage generation entry points:

```bash
python four_stage_generation/single_step/generate_single_step_four_stage_dataset.py --help
python four_stage_generation/multi_step/run_multi_step_four_stage_pipeline.py --help
```

QA generation entry points:

```bash
python qa_generation/qa_generators/stage_one_qa_generator/generate_stage_one_qa_parallel.py --help
INPUT_ROOT=<FINAL_PLAN_ITEM_DIR> OUTPUT_DIR=<QA_OUTPUT_DIR> \
bash qa_generation/qa_generators/stage_two_qa_generator/run_stage_two_qa_generation.sh
```

QA filtering entry points:

```bash
python qa_filtering/score_existing_qa_qwen_two_axis.py --help
python qa_filtering/score_existing_qa_gemini_physical_logic.py --help
python qa_filtering/filter_existing_qa_physical_logic_audit.py --help
```

The Qwen filter evaluates accurate visual grounding and general logical coherence in one pass. The Gemini physical-logic filter evaluates precondition validity, causal dependency, state transition, timeline consistency, and physical feasibility. Both filters fail closed for malformed rows, missing evidence, parser errors, and judge runtime errors.

<p align="center">
  <img src="assets/resource_statistics.png" alt="Resource statistics" width="92%">
</p>

## Causal Trace Generation

The causal trace generator adds task-specific reasoning traces to QA rows. Each trace is checked against task contracts covering visual evidence, preconditions, causal dependencies, physical feasibility, and the reasoning moves expected for the target task family.

```bash
SOURCE_ROOT=<QA_INPUT_DIR> OUTPUT_ROOT=<QA_WITH_TRACES_DIR> \
bash causal_trace_generation/run_task_specific_causal_trace_generation.sh
```

Local trace-contract checks:

```bash
python causal_trace_generation/validate_causal_trace_contracts.py
```

## Training Utilities

Prepare SWIFT-compatible SFT data:

```bash
python sft_training/prepare_swift_sft_dataset.py \
  --input-jsonl <QA_JSONL> \
  --output-jsonl <SWIFT_SFT_JSONL>
```

Run SWIFT SFT through environment-variable configuration:

```bash
cd sft_training
SFT_MODEL=<MODEL_OR_CHECKPOINT> \
SFT_DATASET=<SWIFT_SFT_JSONL> \
bash run_swift_sft.sh
```

Use the RL reward package from Python:

```python
from causal_learner_reward import compute_score
```

The reward package supports rule-based task rewards and optional multimodal rubric judging. See `rl_training/README.md` for the expected reward input schema and judge-backed scoring configuration.

## Repository Layout

| Directory | Purpose |
| --- | --- |
| `evaluation/` | MCQ and open-QA benchmark evaluation without bundled benchmark data. |
| `four_stage_generation/` | Single-step and multi-step plan, localization, keyframe, and atomic-action generation. |
| `qa_generation/` | Active 20-task QA prompt registry and QA generation runners. |
| `qa_filtering/` | Qwen two-axis QA scoring and Gemini physical-logic scoring/auditing. |
| `causal_trace_generation/` | Task-specific causal reasoning trace generation and validation. |
| `sft_training/` | SWIFT SFT data conversion and launch wrapper. |
| `rl_training/` | Causal Learner reward logic for RL training. |

## Data Policy

This repository intentionally excludes benchmark data, training data, raw videos, model checkpoints, generated outputs, runtime caches, machine-specific job scripts, and environment-specific activation commands.

All prompt content required by the active generation, filtering, evaluation, SFT, and reward code is stored in Python or shell source files. Markdown files are kept only as README files.
