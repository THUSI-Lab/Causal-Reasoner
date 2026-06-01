# QA Filtering

This directory contains post-generation QA filtering pipelines. It is separate from QA generation.

The Qwen3.5-397B-A17B filter evaluates each instance in one pass for accurate visual grounding and general logical coherence, then ranks rows by the stricter two-axis score.

The Gemini physical-logic filter evaluates preconditions, causal dependencies, state transitions, timeline consistency, and physical feasibility. It can be run either as a continuous score ranker or as a strict binary audit.

## Files

- `score_existing_qa_qwen_two_axis.py`: command-line entry point for scoring and ranking QA rows.
- `qwen_two_axis_score_core.py`: two-axis Qwen judge rubric, hard-failure caps, and strict parser.
- `score_existing_qa_gemini_physical_logic.py`: command-line entry point for Gemini physical-logic scoring and thresholded ranking.
- `gemini_physical_logic_score_core.py`: five-axis Gemini physical-logic rubric, hard-failure caps, and strict parser.
- `filter_existing_qa_physical_logic_audit.py`: command-line entry point for strict Gemini binary physical-logic auditing.
- `physical_logic_audit_core.py`: binary physical-logic audit rubric and parser.
- `binary_filter_runner.py`: shared runner for binary accept/reject filters.
- `unified_filter_core.py`: shared status normalization utilities used by binary filters.
- `qa_filter_io.py`: JSONL loading, row normalization, source-context extraction, preflight checks, and metadata writing.
- `evidence_frames.py`: deterministic visual evidence extraction for keyframes, clips, prefixes, and clip pairs.
- `judge_api.py`: OpenAI-compatible and Azure OpenAI judge client helper.

For Azure-backed judge calls, set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_PROFILE` or `API_PROFILE`.

## Output

The Qwen scoring script writes `accepted/`, `rejected/`, `qwen_two_axis_scores.jsonl`, and `qwen_two_axis_score_summary.json`.

The Gemini score ranker writes `accepted/`, `rejected/`, `gemini_physical_logic_scores.jsonl`, and `gemini_physical_logic_score_summary.json`.

The Gemini binary audit writes `accepted/`, `rejected/`, `physical_logic_audit_decisions.jsonl`, and `physical_logic_audit_summary.json`.

## Scoring Logic

The Qwen scorer uses two 0-100 axes: visual grounding and logical coherence. The final score is capped by the lower axis and by hard failure tags such as missing visual evidence, visual hallucination, unsupported causal claim, contradiction, wrong step, impossible physics, or timeline mismatch.

The Gemini scorer uses five 0-100 axes: precondition validity, causal dependency, state transition, timeline consistency, and physical feasibility. Rows are accepted only when they pass both rank-based selection and score thresholds.

Both score-based filters fail closed for invalid input rows, missing required evidence, parser errors, and judge runtime errors.
