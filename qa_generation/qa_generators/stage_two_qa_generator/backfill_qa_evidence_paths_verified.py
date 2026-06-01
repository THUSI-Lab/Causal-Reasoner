


import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill")






_plan_cache: Dict[str, Optional[Dict]] = {}


def _load_plan(item_dir: str) -> Optional[Dict]:
    if item_dir in _plan_cache:
        return _plan_cache[item_dir]
    fp = os.path.join(item_dir, "final_plan.json")
    if not os.path.isfile(fp):
        _plan_cache[item_dir] = None
        return None
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

        for s in self.steps:
            sid = int(s["step_id"])
            sg = s.get("step_goal", "")
            cfs = s.get("critical_frames") or []
            cf_list = []
            for j, cf in enumerate(cfs):
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
        if os.path.isfile(c):
            return c
    if plan:
        sv = plan.get("source_video", "")
        if sv and os.path.isfile(sv):
            return sv
    return None


def _resolve_video_clip(item_dir: str, step_id: int) -> Optional[str]:

    nn = f"{step_id:02d}"

    if step_id == 1:
        p = os.path.join(item_dir, "last_frame_segments", "segment_start_to_step01.mp4")
        if os.path.isfile(p):
            return p
    else:
        prev = f"{step_id - 1:02d}"
        p = os.path.join(item_dir, "last_frame_segments", f"segment_step{prev}_to_step{nn}.mp4")
        if os.path.isfile(p):
            return p


    seg_json = os.path.join(item_dir, "stage2", "step_segments.json")
    if os.path.isfile(seg_json):
        try:
            with open(seg_json, "r") as f:
                seg_data = json.load(f)
            for seg in (seg_data.get("segments") or []):
                if int(seg.get("step_id", 0)) == step_id:
                    cr = seg.get("clip_relpath", "")
                    if cr:
                        full = os.path.join(item_dir, "stage2", cr)
                        if os.path.isfile(full):
                            return full
        except Exception:
            pass


    for pat_nn in (nn, str(step_id)):
        matches = glob.glob(os.path.join(item_dir, "stage2", "step_clips", f"step{pat_nn}_*.mp4"))
        if matches:
            return matches[0]
    return None


def _resolve_video_prefix_or_clip(item_dir: str, step_id: int, plan: Optional[Dict] = None) -> Optional[str]:
    v = _resolve_video_prefix(item_dir, step_id, plan=None)
    if v:
        return v
    if plan:
        sv = plan.get("source_video", "")
        if sv and os.path.isfile(sv):
            return sv
    return _resolve_video_clip(item_dir, step_id)


def _find_keyframe_image(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:
    nn = f"{step_id:02d}"
    idx_str = f"{frame_index:03d}"
    for base in [os.path.join(item_dir, "stage3"), item_dir]:
        dirs = glob.glob(os.path.join(base, f"{nn}_*"))
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for ext in ("jpg", "jpeg", "png"):

                m = glob.glob(os.path.join(d, f"frame_{idx_str}_ts_*.{ext}"))
                if m:
                    return m[0]
            for ext in ("jpg", "jpeg", "png"):
                m = glob.glob(os.path.join(d, f"frame_{idx_str}_*.{ext}"))
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
            return [], None, False

        video = _resolve_video_clip(item_dir, sid)
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
                asc = (cf.get("action_state_change_description", "") or "").strip()
                cc = cf.get("causal_chain", {}) if isinstance(cf.get("causal_chain"), dict) else {}

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
            sid0 = idx.goal_to_sid.get(pg)
            sid1 = idx.goal_to_sid.get(ng)
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






def process_task_file(task_jsonl: str, dry_run: bool = False) -> Dict[str, int]:
    task_name = os.path.basename(os.path.dirname(task_jsonl))
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
            log.debug("Error resolving %s/%s: %s", task_name, rec.get("id", "?"), e)
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
        tmp_path = task_jsonl + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in updated:
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, task_jsonl)

    return stats






def main():
    parser = argparse.ArgumentParser(description="Backfill evidence paths (deterministic, rule-based)")
    parser.add_argument("--qa-dir", required=True, nargs="+",
                        help="QA output directories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats only, don't write files")
    args = parser.parse_args()

    all_files = []
    for qa_dir in args.qa_dir:
        for task_dir in sorted(glob.glob(os.path.join(qa_dir, "Task_*"))):
            jsonl = os.path.join(task_dir, "data.jsonl")
            if os.path.isfile(jsonl):
                all_files.append(jsonl)

    if not all_files:
        log.error("No data.jsonl files found")
        sys.exit(1)

    log.info("Processing %d task files%s", len(all_files), " (DRY RUN)" if args.dry_run else "")

    grand = {"total": 0, "filled": 0, "missing": 0, "already_has": 0, "errors": 0}

    for jsonl in all_files:
        task_name = os.path.basename(os.path.dirname(jsonl))
        qa_dir_name = os.path.basename(os.path.dirname(os.path.dirname(jsonl)))
        stats = process_task_file(jsonl, dry_run=args.dry_run)
        for k in grand:
            grand[k] += stats[k]

        fill_rate = stats["filled"] / max(stats["total"], 1) * 100
        marker = "✅" if stats["missing"] == 0 else "⚠️"
        log.info("%s %s/%s: %d total, %d filled (%.1f%%), %d missing, %d errors",
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

    if grand["missing"] > grand["total"] * 0.05:
        log.error("Missing rate > 5%%, please investigate")
        sys.exit(1)


if __name__ == "__main__":
    main()
