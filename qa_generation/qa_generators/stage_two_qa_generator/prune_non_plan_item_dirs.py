


from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


PLAN_FILENAME = "final_plan.json"


@dataclass
class PruneReport:
    root: str
    kept: List[str]
    to_delete: List[str]
    deleted: List[str]
    failed: List[Dict[str, str]]
    skipped: List[str]
    dry_run: bool


def _iter_child_dirs(root: str, *, include_hidden: bool) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        if not include_hidden and name.startswith("."):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append((name, path))
    return out


def _classify_child_dirs(root: str, *, include_hidden: bool) -> Tuple[List[str], List[str], List[str]]:
    kept: List[str] = []
    to_delete: List[str] = []
    skipped: List[str] = []
    for name, path in _iter_child_dirs(root, include_hidden=include_hidden):
        if os.path.islink(path):
            skipped.append(name)
            continue
        plan_path = os.path.join(path, PLAN_FILENAME)
        if os.path.isfile(plan_path):
            kept.append(name)
        else:
            to_delete.append(name)
    return kept, to_delete, skipped


def _confirm_delete(root: str, count: int) -> bool:
    prompt = (
        f"About to permanently delete {count} directories under:\n"
        f"  {root}\n\n"
        'Type "DELETE" to continue: '
    )
    try:
        ans = input(prompt).strip()
    except EOFError:
        return False
    return ans == "DELETE"


def main() -> None:
    p = argparse.ArgumentParser(description="Delete immediate child directories that do not contain final_plan.json.")
    p.add_argument("--root", required=True, help="Root directory containing many video item subfolders.")
    p.add_argument("--delete", action="store_true", help="Actually delete directories (default: dry-run).")
    p.add_argument("--yes", action="store_true", help='Skip interactive confirmation when using --delete.')
    p.add_argument("--force", action="store_true", help="Allow deletion even if 0 valid item dirs are found (destructive).")
    p.add_argument("--include-hidden", action="store_true", help="Include hidden child directories (names starting with '.').")
    p.add_argument("--report-json", default="", help="Optional: write a JSON report to this path.")
    p.add_argument("--print-all", action="store_true", help="Print full kept/to_delete lists (can be long).")
    args = p.parse_args()

    root = os.path.abspath(os.path.expanduser(str(args.root)))
    if not os.path.isdir(root):
        raise SystemExit(f"--root must be an existing directory: {root}")

    kept, to_delete, skipped = _classify_child_dirs(root, include_hidden=bool(args.include_hidden))

    def _preview(items: List[str], label: str) -> None:
        n = len(items)
        if n == 0:
            return
        if bool(args.print_all) or n <= 20:
            print(f"{label} ({n}): {items}")
        else:
            print(f"{label} ({n}): {items[:20]} ... (pass --print-all to show all)")

    print(f"ROOT: {root}")
    print(f"Found child dirs: {len(kept) + len(to_delete) + len(skipped)}")
    print(f"Keep (has {PLAN_FILENAME}): {len(kept)}")
    print(f"Delete (missing {PLAN_FILENAME}): {len(to_delete)}")
    print(f"Skipped (symlink dirs): {len(skipped)}")
    _preview(kept, "KEPT")
    _preview(to_delete, "TO_DELETE")
    _preview(skipped, "SKIPPED")

    dry_run = not bool(args.delete)

    if not to_delete:
        report = PruneReport(
            root=root,
            kept=kept,
            to_delete=[],
            deleted=[],
            failed=[],
            skipped=skipped,
            dry_run=dry_run,
        )
        if args.report_json:
            _write_report(report, str(args.report_json))
        print("Nothing to delete.")
        return

    if not kept and not bool(args.force):
        raise SystemExit(
            f"Safety stop: 0 child dirs contain {PLAN_FILENAME}. Refusing to delete anything.\n"
            f"Double-check --root. If you really want to delete all child dirs under {root}, re-run with --force."
        )

    if dry_run:
        report = PruneReport(
            root=root,
            kept=kept,
            to_delete=to_delete,
            deleted=[],
            failed=[],
            skipped=skipped,
            dry_run=True,
        )
        if args.report_json:
            _write_report(report, str(args.report_json))
        print("DRY RUN: no directories were deleted. Re-run with --delete (and optionally --yes).")
        return

    if not bool(args.yes):
        if not _confirm_delete(root, len(to_delete)):
            raise SystemExit("Aborted (confirmation not received).")

    deleted: List[str] = []
    failed: List[Dict[str, str]] = []
    for name in to_delete:
        path = os.path.join(root, name)
        try:
            shutil.rmtree(path)
            deleted.append(name)
        except Exception as e:
            failed.append({"dir": name, "error": str(e)})

    report = PruneReport(
        root=root,
        kept=kept,
        to_delete=to_delete,
        deleted=deleted,
        failed=failed,
        skipped=skipped,
        dry_run=False,
    )
    if args.report_json:
        _write_report(report, str(args.report_json))

    print(f"Deleted: {len(deleted)}/{len(to_delete)}")
    if failed:
        print(f"Failed: {len(failed)} (see report or stderr)")
        for f in failed[:20]:
            print(f"  FAIL dir={f.get('dir')} err={f.get('error')}", file=sys.stderr)
        raise SystemExit(2)


def _write_report(report: PruneReport, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Wrote report: {path}")


if __name__ == "__main__":
    main()
