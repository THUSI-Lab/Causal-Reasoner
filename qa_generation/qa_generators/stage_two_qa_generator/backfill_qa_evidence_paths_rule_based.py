


import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_fast")





_plan_cache: Dict[str, Optional[Dict]] = {}


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


class PlanIndex:
    def __init__(self, plan: Dict):
        self.plan = plan
        self.steps = _sorted_steps(plan)
        self.last_step_id = int(self.steps[-1]["step_id"]) if self.steps else 0
        self.first_step_id = int(self.steps[0]["step_id"]) if self.steps else 0
        self.goal_to_sid: Dict[str, int] = {}
        for s in self.steps:
            self.goal_to_sid[s.get("step_goal", "")] = int(s["step_id"])
        self.keyframe_lookup: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
        self.asc_lookup: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.cf_by_step: Dict[int, List[Tuple[Dict, int]]] = {}


        self.sid_to_goal: Dict[int, str] = {}

        for s in self.steps:
            sid = int(s["step_id"])
            sg = s.get("step_goal", "")
            self.sid_to_goal[sid] = sg
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






def _construct_video_prefix(item_dir: str, step_id: int) -> str:

    nn = f"{step_id:02d}"
    return os.path.join(item_dir, "cumulative_last_frame_segments",
                        f"segment_start_to_step{nn}_last.mp4")


def _construct_video_clip(item_dir: str, step_id: int, step_goal: str = "") -> str:

    nn = f"{step_id:02d}"



    if step_goal:

        slug = step_goal.lower().strip()

        slug = slug.replace("'", "").replace('"', '').replace(",", "")
        slug = "_".join(slug.split())

        return os.path.join(item_dir, "stage2", "step_clips",
                            f"step{nn}_{slug}.mp4")
    return os.path.join(item_dir, "stage2", "step_clips", f"step{nn}.mp4")


def _construct_keyframe_path(item_dir: str, step_id: int, frame_index: int,
                             step_goal: str = "", timestamp: str = "") -> str:

    nn = f"{step_id:02d}"
    idx_str = f"{frame_index:03d}"

    if step_goal:
        slug = step_goal.lower().strip()
        slug = slug.replace("'", "").replace('"', '').replace(",", "")
        slug = "_".join(slug.split())
        dir_name = f"{nn}_{slug}"
    else:
        dir_name = f"{nn}_unknown"
    if timestamp:
        return os.path.join(item_dir, "stage3", dir_name,
                            f"frame_{idx_str}_ts_{timestamp}.jpg")
    return os.path.join(item_dir, "stage3", dir_name, f"frame_{idx_str}.jpg")







_dir_listing_cache: Dict[str, List[str]] = {}


def _list_dir_cached(dirpath: str) -> List[str]:

    if dirpath in _dir_listing_cache:
        return _dir_listing_cache[dirpath]
    try:
        entries = os.listdir(dirpath)
        _dir_listing_cache[dirpath] = entries
        return entries
    except Exception:
        _dir_listing_cache[dirpath] = []
        return []


def _find_video_prefix(item_dir: str, step_id: int, plan: Optional[Dict] = None) -> Optional[str]:
    nn = f"{step_id:02d}"
    cum_dir = os.path.join(item_dir, "cumulative_last_frame_segments")
    entries = _list_dir_cached(cum_dir)


    for pat in [f"segment_start_to_step{nn}_last.mp4",
                f"segment_start_to_step{nn}.mp4"]:
        if pat in entries:
            return os.path.join(cum_dir, pat)


    nested = os.path.join(cum_dir, "cumulative_last_frame_segments")
    nested_entries = _list_dir_cached(nested)
    for pat in [f"segment_start_to_step{nn}_last.mp4",
                f"segment_start_to_step{nn}.mp4"]:
        if pat in nested_entries:
            return os.path.join(nested, pat)


    prefix_dir = os.path.join(item_dir, "prefix_clips")
    pf_entries = _list_dir_cached(prefix_dir)
    for pat in [f"prefix_step{nn}.mp4", f"prefix_step{step_id}.mp4"]:
        if pat in pf_entries:
            return os.path.join(prefix_dir, pat)


    if plan:
        sv = plan.get("source_video", "")
        if sv:
            return sv

    return None


def _find_video_clip(item_dir: str, step_id: int) -> Optional[str]:
    nn = f"{step_id:02d}"


    lfs_dir = os.path.join(item_dir, "last_frame_segments")
    lfs_entries = _list_dir_cached(lfs_dir)
    if step_id == 1:
        pat = "segment_start_to_step01.mp4"
        if pat in lfs_entries:
            return os.path.join(lfs_dir, pat)
    else:
        prev = f"{step_id - 1:02d}"
        pat = f"segment_step{prev}_to_step{nn}.mp4"
        if pat in lfs_entries:
            return os.path.join(lfs_dir, pat)


    clips_dir = os.path.join(item_dir, "stage2", "step_clips")
    clips_entries = _list_dir_cached(clips_dir)
    for fn in clips_entries:
        if fn.startswith(f"step{nn}_") and fn.endswith(".mp4"):
            return os.path.join(clips_dir, fn)

    n_str = str(step_id)
    if n_str != nn:
        for fn in clips_entries:
            if fn.startswith(f"step{n_str}_") and fn.endswith(".mp4"):
                return os.path.join(clips_dir, fn)

    return None


def _find_keyframe(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:
    nn = f"{step_id:02d}"
    idx_str = f"{frame_index:03d}"

    for base in [os.path.join(item_dir, "stage3"), item_dir]:
        base_entries = _list_dir_cached(base)
        for dn in base_entries:
            if dn.startswith(f"{nn}_"):
                full_d = os.path.join(base, dn)
                d_entries = _list_dir_cached(full_d)

                for fn in d_entries:
                    if fn.startswith(f"frame_{idx_str}_ts_") and fn.endswith((".jpg", ".jpeg", ".png")):
                        return os.path.join(full_d, fn)

                for fn in d_entries:
                    if fn.startswith(f"frame_{idx_str}_") and fn.endswith((".jpg", ".jpeg", ".png")):
                        return os.path.join(full_d, fn)
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
                return [], None, False
        else:
            sid = idx.last_step_id
        video = _find_video_prefix(item_dir, sid, plan)
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
            return [], None, False
        video = _find_video_clip(item_dir, sid)
        return [], video, video is not None


    if et == "keyframe_single":
        sg = llm.get("step_goal", "")
        sid = idx.goal_to_sid.get(sg)
        if sid is None:
            return [], None, False

        patient = llm.get("patient", "").strip()
        mechanism = llm.get("mechanism", "").strip()
        if patient and mechanism:
            key = (sg, patient, mechanism)
            if key in idx.keyframe_lookup:
                sid_k, fi = idx.keyframe_lookup[key]
                img = _find_keyframe(item_dir, sid_k, fi)
                if img:
                    return [img], None, True

        cf_list = idx.cf_by_step.get(sid, [])

        if len(cf_list) == 1:
            _, fi = cf_list[0]
            img = _find_keyframe(item_dir, sid, fi)
            if img:
                return [img], None, True

        if len(cf_list) >= 2:

            for cf, fi in cf_list:
                asc = (cf.get("action_state_change_description", "") or "").strip()
                if asc:
                    answer = rec.get("conversations", [{}])[-1].get("value", "") if rec.get("conversations") else ""
                    if asc[:50] in answer:
                        img = _find_keyframe(item_dir, sid, fi)
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
                if sp and sp == str(cc.get("spatial_preconditions", "")):
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if ap and ap == str(cc.get("affordance_preconditions", "")):
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if se and se == str(cc.get("spatial_effects", "")):
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if ae and ae == str(cc.get("affordance_effects", "")):
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True


            spc = llm.get("selected_pre_clause", "")
            sec = llm.get("selected_eff_clause", "")
            for cf, fi in cf_list:
                cf_text = json.dumps(cf, ensure_ascii=False)
                if spc and spc in cf_text:
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True
                if sec and sec in cf_text:
                    img = _find_keyframe(item_dir, sid, fi)
                    if img:
                        return [img], None, True


            _, fi_first = cf_list[0]
            img = _find_keyframe(item_dir, sid, fi_first)
            if img:
                return [img], None, True

        return [], None, False


    if et == "video_clip_pair":
        if task.startswith("Task_14"):
            pg = llm.get("prev_step_goal", "")
            ng = llm.get("next_step_goal", "")
            sid0 = idx.goal_to_sid.get(pg)
            sid1 = idx.goal_to_sid.get(ng)
            if sid0 is None or sid1 is None:
                return [], None, False
            c0 = _find_video_clip(item_dir, sid0)
            c1 = _find_video_clip(item_dir, sid1)
            if c0 and c1:
                return [], json.dumps([c0, c1]), True
            return [], None, False

        if task.startswith("Task_16"):
            sid0 = idx.first_step_id
            sid1 = idx.last_step_id
            c0 = _find_video_clip(item_dir, sid0)
            c1 = _find_video_clip(item_dir, sid1)
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
    parser = argparse.ArgumentParser(description="Fast backfill with dir listing cache")
    parser.add_argument("--qa-dir", required=True, nargs="+")
    parser.add_argument("--output-dir", default=None, help="Output base dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inplace", action="store_true", help="Write back to original files")
    args = parser.parse_args()

    all_files = []
    for qa_dir in args.qa_dir:
        for task_dir in sorted(glob.glob(os.path.join(qa_dir, "Task_*"))):
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
    log.info("SUMMARY: %d total, %d filled (%.1f%%), %d already_had, %d missing (%.1f%%), %d errors",
             grand["total"], grand["filled"],
             grand["filled"] / max(grand["total"], 1) * 100,
             grand["already_has"],
             grand["missing"], grand["missing"] / max(grand["total"], 1) * 100,
             grand["errors"])
    log.info("Plans loaded: %d", len(_plan_cache))
    log.info("Dir listings cached: %d", len(_dir_listing_cache))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
