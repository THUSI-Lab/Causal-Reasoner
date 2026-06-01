from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from .binary_filter_runner import run_binary_filter
    from .physical_logic_audit_core import (
        build_physical_logic_audit_messages,
        parse_physical_logic_audit_response,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from binary_filter_runner import run_binary_filter
    from physical_logic_audit_core import build_physical_logic_audit_messages, parse_physical_logic_audit_response


AUDIT_NAME = "gemini_physical_logic_binary_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing QA rows for strict physical logic.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Input QA dir containing Task_*/data.jsonl.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dir for accepted/, rejected/, decisions, and summary.")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1100)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "Gemini-3.0-Pro"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("JUDGE_REASONING_EFFORT", "medium"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--allow-partial-source-context", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_binary_filter(
        args,
        filter_name=AUDIT_NAME,
        metadata_key="physical_logic_audit",
        kind="physical",
        build_messages=build_physical_logic_audit_messages,
        parse_response=parse_physical_logic_audit_response,
        decisions_filename="physical_logic_audit_decisions.jsonl",
        summary_filename="physical_logic_audit_summary.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
