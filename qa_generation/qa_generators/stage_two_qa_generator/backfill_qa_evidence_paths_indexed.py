


import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_qa_evidence_paths_indexed")





class CachedFS:


    def __init__(self):
        self.files: Set[str] = set()

        self.dir_contents: Dict[str, List[str]] = defaultdict(list)

    def load_index(self, index_path: str):
        with open(index_path, "r") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                self.files.add(p)
                d = os.path.dirname(p)
                self.dir_contents[d].append(os.path.basename(p))
        log.info("Loaded %d files from %s (%d dirs)", len(self.files), index_path, len(self.dir_contents))

    def isfile(self, path: str) -> bool:
        return path in self.files

    def glob(self, pattern: str) -> List[str]:

        d = os.path.dirname(pattern)
        base_pat = os.path.basename(pattern)
        if d not in self.dir_contents:
            return []

        regex = re.compile("^" + re.escape(base_pat).replace(r"\*", ".*") + "$")
        results = []
        for fn in self.dir_contents[d]:
            if regex.match(fn):
                results.append(os.path.join(d, fn))
        return results

    def glob_dirs(self, pattern: str) -> List[str]:

        d = os.path.dirname(pattern)
        base_pat = os.path.basename(pattern)
        regex = re.compile("^" + re.escape(base_pat).replace(r"\*", ".*") + "$")
        results = set()

        prefix = d + "/"
        for known_dir in self.dir_contents:
            if known_dir.startswith(prefix):
                rel = known_dir[len(prefix):]

                if "/" not in rel and regex.match(rel):
                    results.add(known_dir)
        return sorted(results)


FS = CachedFS()





_plan_cache: Dict[str, Optional[Dict]] = {}


def _preload_plans_from_json(plan_json_path: str):

    log.info("Loading plans from %s ...", plan_json_path)
    with open(plan_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item_dir, plan in data.items():
        _plan_cache[item_dir] = plan
    log.info("Pre-loaded %d plans from JSON", len(data))


def _preload_plans(plan_index_paths: List[str]):
    count = 0
    for idx_path in plan_index_paths:
        with open(idx_path, "r") as f:
            for line in f:
                fp = line.strip()
                if not fp:
                    continue
                item_dir = os.path.dirname(fp)
                if item_dir in _plan_cache:
                    continue
                try:
                    with open(fp, "r", encoding="utf-8") as pf:
                        _plan_cache[item_dir] = json.load(pf)
                    count += 1
                except Exception:
                    _plan_cache[item_dir] = None
    log.info("Pre-loaded %d plans", count)


def _load_plan(item_dir: str) -> Optional[Dict]:
    if item_dir in _plan_cache:
        return _plan_cache[item_dir]

    fp = os.path.join(item_dir, "final_plan.json")
    try:
        with open(fp, "r", encoding="utf-8") as f:
            plan = json.load(f)
        _plan_cache[item_dir] = plan
        return plan
    except Exception:
        _plan_cache[item_dir] = None
        return None


def _sorted_steps(plan: Dict) -> List[Dict]:
    steps = plan.get("steps") or []
    return sorted(steps, key=lambda s: int(s.get("step_id", 0) or 0))


def _normalize_goal(s: str) -> str:

    if not s:
        return ""

    s = s.replace('"', "'")
    s = s.replace('\u201c', "'").replace('\u201d', "'")                       
    s = s.replace('\u2018', "'").replace('\u2019', "'")                       

    s = " ".join(s.split())
    return s


class PlanIndex:
    def __init__(self, plan: Dict):
        self.plan = plan
        self.steps = _sorted_steps(plan)
        self.last_step_id = int(self.steps[-1]["step_id"]) if self.steps else 0
        self.first_step_id = int(self.steps[0]["step_id"]) if self.steps else 0
        self.goal_to_sid: Dict[str, int] = {}
        for s in self.steps:
            self.goal_to_sid[s.get("step_goal", "")] = int(s["step_id"])


        self.goal_to_sid_norm: Dict[str, int] = {}
        for s in self.steps:
            g = s.get("step_goal", "")
            self.goal_to_sid_norm[_normalize_goal(g)] = int(s["step_id"])
        self.keyframe_lookup: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
        self.asc_lookup: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.cf_by_step: Dict[int, List[Tuple[Dict, int]]] = {}
        for s in self.steps:
            sid = int(s["step_id"])
            sg = s.get("step_goal", "")
            cfs = s.get("critical_frames") or []
            cf_list = []
            for cf in cfs:
                if not isinstance(cf, dict):
                    continue
                fi = cf.get("frame_index")
                if fi is None:
                    continue
                fi = int(fi)
                cf_list.append((cf, fi))
                intr = cf.get("interaction", {})
                if not isinstance(intr, dict):
                    intr = {}
                hotspot = intr.get("hotspot", intr)
                if not isinstance(hotspot, dict):
                    hotspot = {}
                patient = (hotspot.get("patient", "") or hotspot.get("description", "")).strip()
                mechanism = (hotspot.get("mechanism", "") or "").strip()
                asc = (cf.get("action_state_change_description", "") or "").strip()
                if patient and mechanism:
                    self.keyframe_lookup[(sg, patient, mechanism)] = (sid, fi)
                if asc:
                    self.asc_lookup[(sg, asc)] = (sid, fi)
            self.cf_by_step[sid] = cf_list


_index_cache: Dict[str, Optional[PlanIndex]] = {}


def _get_index(item_dir: str) -> Optional[PlanIndex]:
    if item_dir in _index_cache:
        return _index_cache[item_dir]
    plan = _load_plan(item_dir)
    if plan is None:
        _index_cache[item_dir] = None
        return None
    idx = PlanIndex(plan)
    _index_cache[item_dir] = idx
    return idx






def _resolve_video_prefix(item_dir: str, step_id: int, plan: Optional[Dict] = None) -> Optional[str]:
    nn = f"{step_id:02d}"
    n = str(step_id)
    candidates = [
        os.path.join(item_dir, "prefix_clips", f"prefix_step{nn}.mp4"),
        os.path.join(item_dir, "prefix_clips", f"prefix_step{n}.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{nn}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{nn}.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", "cumulative_last_frame_segments",
                     f"segment_start_to_step{nn}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", "cumulative_last_frame_segments",
                     f"segment_start_to_step{nn}.mp4"),
    ]
    for c in candidates:
        if FS.isfile(c):
            return c
    if plan:
        sv = plan.get("source_video", "")
        if sv:
            return sv
    return None


def _resolve_video_clip(item_dir: str, step_id: int) -> Optional[str]:
    nn = f"{step_id:02d}"
    if step_id == 1:
        p = os.path.join(item_dir, "last_frame_segments", "segment_start_to_step01.mp4")
        if FS.isfile(p):
            return p
    else:
        prev = f"{step_id - 1:02d}"
        p = os.path.join(item_dir, "last_frame_segments", f"segment_step{prev}_to_step{nn}.mp4")
        if FS.isfile(p):
            return p


    for pat_nn in (nn, str(step_id)):
        matches = FS.glob(os.path.join(item_dir, "stage2", "step_clips", f"step{pat_nn}_*.mp4"))
        if matches:
            return matches[0]



    seg_data = _load_step_segments(item_dir)
    if seg_data:
        for seg in (seg_data.get("segments") or []):
            if int(seg.get("step_id", 0)) == step_id:
                cr = seg.get("clip_relpath", "")
                if cr:
                    return os.path.join(item_dir, "stage2", cr)



    idx = _get_index(item_dir)
    if idx and idx.plan:
        sv = idx.plan.get("source_video", "")
        if sv:
            return sv

    return None


_step_seg_cache: Dict[str, Optional[Dict]] = {}


def _preload_step_segments(seg_json_path: str):

    log.info("Loading step_segments from %s ...", seg_json_path)
    with open(seg_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item_dir, clips in data.items():

        segments = [{"step_id": int(sid), "clip_relpath": cr} for sid, cr in clips.items()]
        _step_seg_cache[item_dir] = {"segments": segments}
    log.info("Pre-loaded %d step_segments from JSON", len(data))


def _load_step_segments(item_dir: str) -> Optional[Dict]:
    if item_dir in _step_seg_cache:
        return _step_seg_cache[item_dir]


    return None


def _find_keyframe_image(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:
    nn = f"{step_id:02d}"
    idx_str = f"{frame_index:03d}"
    for base in [os.path.join(item_dir, "stage3"), item_dir]:
        dirs = FS.glob_dirs(os.path.join(base, f"{nn}_*"))
        for d in dirs:
            for ext in ("jpg", "jpeg", "png"):
                m = FS.glob(os.path.join(d, f"frame_{idx_str}_ts_*.{ext}"))
                if m:
                    return m[0]
            for ext in ("jpg", "jpeg", "png"):
                m = FS.glob(os.path.join(d, f"frame_{idx_str}_*.{ext}"))
                if m:
                    return m[0]
    return None






def resolve_evidence(rec: Dict) -> Tuple[List[str], Optional[str], bool]:
    meta = rec.get("meta", {})
    et = meta.get("evidence_type", "")
    item_dir = meta.get("item_dir", "")
    llm = meta.get("llm_fields", {})
    task = meta.get("task_name", "")

    if not item_dir or not et:
        return [], None, False

    idx = _get_index(item_dir)
    if idx is None:
        return [], None, False

    plan = idx.plan


    if et == "video_prefix":
        if task.startswith("Task_08") or task.startswith("Task_09"):
            sid = idx.last_step_id
        elif "prefix_end_step_id" in llm:
            sid = int(llm["prefix_end_step_id"])
        elif "last_completed_step_goal" in llm:
            g = llm["last_completed_step_goal"]
            sid = idx.goal_to_sid.get(g)
            if sid is None:
                sid = idx.goal_to_sid_norm.get(_normalize_goal(g))
            if sid is None:
                return [], None, False
        else:
            sid = idx.last_step_id
        video = _resolve_video_prefix(item_dir, sid, plan)
        return [], video, video is not None


    if et == "video_clip":
        sid = None
        if "step_id" in llm:
            try:
                sid = int(llm["step_id"])
            except (ValueError, TypeError):
                pass
        if sid is None and "step_goal" in llm:
            sid = idx.goal_to_sid.get(llm["step_goal"])
            if sid is None:
                sid = idx.goal_to_sid_norm.get(_normalize_goal(llm["step_goal"]))
        if sid is None:
            return [], None, False
        video = _resolve_video_clip(item_dir, sid)
        return [], video, video is not None


    if et == "keyframe_single":
        sg = llm.get("step_goal", "")
        sid = idx.goal_to_sid.get(sg)
        if sid is None:
            sid = idx.goal_to_sid_norm.get(_normalize_goal(sg))
            if sid is not None:

                for _g, _sid in idx.goal_to_sid.items():
                    if _sid == sid:
                        sg = _g
                        break
        if sid is None:
            return [], None, False

        patient = llm.get("patient", "").strip()
        mechanism = llm.get("mechanism", "").strip()
        if patient and mechanism:
            key = (sg, patient, mechanism)
            if key in idx.keyframe_lookup:
                sid_k, fi = idx.keyframe_lookup[key]
                img = _find_keyframe_image(item_dir, sid_k, fi)
                if img:
                    return [img], None, True

        cf_list = idx.cf_by_step.get(sid, [])

        if len(cf_list) == 1:
            _, fi = cf_list[0]
            img = _find_keyframe_image(item_dir, sid, fi)
            if img:
                return [img], None, True

        if len(cf_list) == 2:
            for cf, fi in cf_list:
                asc = (cf.get("action_state_change_description", "") or "").strip()
                if asc:
                    key_asc = (sg, asc)
                    if key_asc in idx.asc_lookup:
                        _, fi_match = idx.asc_lookup[key_asc]
                        answer = rec.get("conversations", [{}])[-1].get("value", "") if rec.get("conversations") else ""
                        if asc[:50] in answer:
                            img = _find_keyframe_image(item_dir, sid, fi_match)
                            if img:
                                return [img], None, True

            sp = llm.get("spatial_preconditions", "")
            ap = llm.get("affordance_preconditions", "")
            se = llm.get("spatial_effects", "")
            ae = llm.get("affordance_effects", "")
            for cf, fi in cf_list:
                cc = cf.get("causal_chain", {})
                if not isinstance(cc, dict):
                    continue
                cf_sp = str(cc.get("spatial_preconditions", ""))
                cf_ap = str(cc.get("affordance_preconditions", ""))
                cf_se = str(cc.get("spatial_effects", ""))
                cf_ae = str(cc.get("affordance_effects", ""))
                if sp and sp == cf_sp:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if ap and ap == cf_ap:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if se and se == cf_se:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if ae and ae == cf_ae:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True

            spc = llm.get("selected_pre_clause", "")
            sec = llm.get("selected_eff_clause", "")
            for cf, fi in cf_list:
                cf_text = json.dumps(cf, ensure_ascii=False)
                if spc and spc in cf_text:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if sec and sec in cf_text:
                    img = _find_keyframe_image(item_dir, sid, fi)
                    if img:
                        return [img], None, True

            _, fi_first = cf_list[0]
            img = _find_keyframe_image(item_dir, sid, fi_first)
            if img:
                return [img], None, True

        return [], None, False


    if et == "video_clip_pair":
        if task.startswith("Task_14"):
            pg = llm.get("prev_step_goal", "")
            ng = llm.get("next_step_goal", "")
            sid0 = idx.goal_to_sid.get(pg) or idx.goal_to_sid_norm.get(_normalize_goal(pg))
            sid1 = idx.goal_to_sid.get(ng) or idx.goal_to_sid_norm.get(_normalize_goal(ng))
            if sid0 is None or sid1 is None:
                return [], None, False
            c0 = _resolve_video_clip(item_dir, sid0)
            c1 = _resolve_video_clip(item_dir, sid1)
            if c0 and c1:
                return [], json.dumps([c0, c1]), True
            return [], None, False

        if task.startswith("Task_16"):
            sid0 = idx.first_step_id
            sid1 = idx.last_step_id
            c0 = _resolve_video_clip(item_dir, sid0)
            c1 = _resolve_video_clip(item_dir, sid1)
            if c0 and c1:
                return [], json.dumps([c0, c1]), True
            return [], None, False

    return [], None, False






def process_task_file(task_jsonl: str, output_jsonl: str, dry_run: bool = False) -> Dict[str, int]:
    stats = {"total": 0, "filled": 0, "missing": 0, "already_has": 0, "errors": 0}

    records = []
    with open(task_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                stats["errors"] += 1
                records.append(None)

    stats["total"] = len(records)

    updated = []
    for rec in records:
        if rec is None:
            updated.append(None)
            continue

        existing_images = rec.get("image", [])
        if existing_images and any(p for p in existing_images if p):
            stats["already_has"] += 1
            updated.append(rec)
            continue

        try:
            images, video, success = resolve_evidence(rec)
        except Exception as e:
            log.debug("Error resolving %s: %s", rec.get("id", "?"), e)
            stats["errors"] += 1
            updated.append(rec)
            continue

        if success and (images or video):
            rec["image"] = images
            ev_files = list(images)
            if video:
                try:
                    parsed = json.loads(video)
                    if isinstance(parsed, list):
                        ev_files.extend(parsed)
                    else:
                        ev_files.append(video)
                except (json.JSONDecodeError, TypeError):
                    ev_files.append(video)
            rec["meta"]["evidence_files"] = ev_files
            if video:
                rec["meta"]["video"] = video
                rec["video"] = video
            stats["filled"] += 1
        else:
            stats["missing"] += 1

        updated.append(rec)

    if not dry_run:
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for rec in updated:
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats






def main():
    parser = argparse.ArgumentParser(description="Fast cached backfill of evidence paths")
    parser.add_argument("--qa-dir", required=True, nargs="+", help="QA output directories")
    parser.add_argument("--file-index", required=True, nargs="+", help="Pre-built file index from find")
    parser.add_argument("--plan-index", nargs="+", default=[], help="Pre-built plan file index")
    parser.add_argument("--plan-json", default=None, help="Pre-dumped plans JSON (item_dir -> plan)")
    parser.add_argument("--seg-json", default=None, help="Pre-dumped step_segments JSON")
    parser.add_argument("--output-dir", default=None, help="Output base dir (default: writes _filled suffix)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inplace", action="store_true", help="Write back to original files")
    args = parser.parse_args()


    for idx_path in args.file_index:
        FS.load_index(idx_path)


    if args.plan_json:
        _preload_plans_from_json(args.plan_json)
    if args.plan_index:
        _preload_plans(args.plan_index)


    if args.seg_json:
        _preload_step_segments(args.seg_json)

    import glob as globmod
    all_files = []
    for qa_dir in args.qa_dir:
        for task_dir in sorted(globmod.glob(os.path.join(qa_dir, "Task_*"))):
            jsonl = os.path.join(task_dir, "data.jsonl")
            if os.path.isfile(jsonl):
                all_files.append((qa_dir, jsonl))

    if not all_files:
        log.error("No data.jsonl files found")
        sys.exit(1)

    log.info("Processing %d task files%s", len(all_files), " (DRY RUN)" if args.dry_run else "")

    grand = {"total": 0, "filled": 0, "missing": 0, "already_has": 0, "errors": 0}

    for qa_dir, jsonl in all_files:
        task_name = os.path.basename(os.path.dirname(jsonl))
        qa_dir_name = os.path.basename(qa_dir)

        if args.inplace:
            output_jsonl = jsonl
        elif args.output_dir:
            output_jsonl = os.path.join(args.output_dir, qa_dir_name, task_name, "data.jsonl")
        else:
            parent = os.path.dirname(qa_dir)
            output_jsonl = os.path.join(parent, qa_dir_name + "_filled", task_name, "data.jsonl")

        stats = process_task_file(jsonl, output_jsonl, dry_run=args.dry_run)
        for k in grand:
            grand[k] += stats[k]

        fill_rate = stats["filled"] / max(stats["total"], 1) * 100
        marker = "OK" if stats["missing"] == 0 else "WARN"
        log.info("[%s] %s/%s: %d total, %d filled (%.1f%%), %d missing, %d err",
                 marker, qa_dir_name, task_name,
                 stats["total"], stats["filled"], fill_rate, stats["missing"], stats["errors"])

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("  Total records:     %d", grand["total"])
    log.info("  Filled:            %d (%.1f%%)", grand["filled"],
             grand["filled"] / max(grand["total"], 1) * 100)
    log.info("  Already had paths: %d", grand["already_has"])
    log.info("  Missing (no file): %d (%.1f%%)", grand["missing"],
             grand["missing"] / max(grand["total"], 1) * 100)
    log.info("  Errors:            %d", grand["errors"])
    log.info("=" * 70)


if __name__ == "__main__":
    main()
