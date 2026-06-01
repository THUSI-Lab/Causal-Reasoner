




from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import generate_stage_two_qa as qa              


_ASCII_SNAKE_FULL_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _ascii_snake_token(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()

    s = re.sub(r"[-\s]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return ""
    if not s[0].isalpha():
        s = "obj_" + s
    s = re.sub(r"_+", "_", s).strip("_")
    if _ASCII_SNAKE_FULL_RE.fullmatch(s):
        return s


    parts = [p for p in re.split(r"_+", s) if p]
    parts2 = [re.sub(r"[^a-z0-9]+", "", p) for p in parts]
    parts2 = [p for p in parts2 if p]
    s2 = "_".join(parts2).strip("_")
    if s2 and not s2[0].isalpha():
        s2 = "obj_" + s2
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2


def _atomic_write_json(path: str, obj: Any) -> None:
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_fix_", suffix=".json", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def _one_line(s: Any, *, max_len: int = 260) -> str:
    t = str(s or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t


def _fix_plan(plan: Any) -> Tuple[bool, List[Dict[str, str]], int, int]:

    if not isinstance(plan, dict):
        return False, [], 0, 0
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return False, [], 0, 0

    changed = False
    changes: List[Dict[str, str]] = []
    patient_n = 0
    aff_n = 0

    for si, st in enumerate(steps):
        if not isinstance(st, dict):
            continue
        cc = st.get("causal_chain")
        if isinstance(cc, dict):
            pat = cc.get("patient")
            if isinstance(pat, str) and pat.strip():
                new_pat = _ascii_snake_token(pat)
                if new_pat and new_pat != pat:
                    cc["patient"] = new_pat
                    changed = True
                    patient_n += 1
                    changes.append({"path": f"steps[{si}].causal_chain.patient", "old": pat, "new": new_pat})

        cfs = st.get("critical_frames")
        if not isinstance(cfs, list):
            continue
        for cj, cf in enumerate(cfs):
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if not isinstance(intr, dict):
                continue
            aff = intr.get("affordance_type")
            if isinstance(aff, str) and aff.strip():
                new_aff = _ascii_snake_token(aff)
                if new_aff and new_aff != aff:
                    intr["affordance_type"] = new_aff
                    changed = True
                    aff_n += 1
                    changes.append(
                        {"path": f"steps[{si}].critical_frames[{cj}].interaction.affordance_type", "old": aff, "new": new_aff}
                    )

    return changed, changes, patient_n, aff_n


def main() -> None:
    p = argparse.ArgumentParser(description="Fix non-ASCII identifier tokens in Mani-LongVideo final_plan.json files (dry-run by default).")
    p.add_argument("--input-root", required=True, help="Dataset root containing many item dirs with final_plan.json.")
    p.add_argument("--apply", action="store_true", help="Write changes in place. Default is dry-run (no writes).")
    p.add_argument(
        "--backup-ext",
        default=".bak",
        help="When --apply is set, write a backup file as <json><ext> (default: .bak). Use '' to disable backups.",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N items (0 = no limit).")
    p.add_argument("--print-first", type=int, default=30, help="Print first N changed items (default: 30).")
    p.add_argument("--report-json", default="", help="Optional path to write a JSON report of all changes.")
    args = p.parse_args()

    input_root = os.path.abspath(str(args.input_root))
    item_dirs = qa._list_item_dirs(input_root)
    if int(args.limit) > 0:
        item_dirs = item_dirs[: int(args.limit)]
    if not item_dirs:
        raise SystemExit(f"No item dirs found under {input_root} (expecting final_plan.json).")

    apply = bool(args.apply)
    backup_ext = str(args.backup_ext or "")
    want_report = bool(str(args.report_json or "").strip())
    max_print = max(0, int(args.print_first))

    total = 0
    changed_items = 0
    patient_changes = 0
    affordance_changes = 0
    first_changed: List[Dict[str, Any]] = []
    all_changed: Optional[List[Dict[str, Any]]] = [] if want_report else None
    read_errors: List[Dict[str, str]] = []

    for item_dir in item_dirs:
        total += 1
        plan_path = os.path.join(item_dir, "final_plan.json")
        rel = os.path.relpath(plan_path, input_root)
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as e:
            read_errors.append({"path": rel, "error": _one_line(e)})
            continue

        changed, changes, p_n, a_n = _fix_plan(plan)
        if not changed:
            continue

        changed_items += 1
        patient_changes += int(p_n)
        affordance_changes += int(a_n)

        row: Dict[str, Any] = {
            "item_dir": os.path.relpath(item_dir, input_root),
            "path": rel,
            "changes": changes,
        }
        if len(first_changed) < max_print:
            first_changed.append(row)
        if all_changed is not None:
            all_changed.append(row)

        if apply:
            if backup_ext:
                bak = plan_path + backup_ext
                if not os.path.exists(bak):
                    os.makedirs(os.path.dirname(os.path.abspath(bak)) or ".", exist_ok=True)
                    shutil.copy2(plan_path, bak)
            _atomic_write_json(plan_path, plan)

    print(f"input_root: {input_root}")
    print(f"items_scanned: {total}")
    print(f"items_with_changes: {changed_items}")
    print(f"patient_changes: {patient_changes}")
    print(f"affordance_type_changes: {affordance_changes}")
    if read_errors:
        print(f"read_errors: {len(read_errors)}")
    print(f"mode: {'APPLY (in-place writes)' if apply else 'DRY-RUN (no writes)'}")

    if first_changed:
        print("\nFirst changed items:")
        for row in first_changed:
            path = str(row.get("path") or "")
            ch = row.get("changes") or []
            if isinstance(ch, list) and ch:
                c0 = ch[0] if isinstance(ch[0], dict) else {}
                msg = f"{c0.get('path')}: {c0.get('old')!r} -> {c0.get('new')!r}"
            else:
                msg = "changed"
            print(f"- {path}: {msg}")

    report_path = str(args.report_json or "").strip()
    if report_path:
        out_path = os.path.abspath(report_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_root": input_root,
                    "apply": apply,
                    "items_scanned": total,
                    "items_with_changes": changed_items,
                    "patient_changes": patient_changes,
                    "affordance_type_changes": affordance_changes,
                    "changed_items": all_changed or [],
                    "read_errors": read_errors,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
        print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
