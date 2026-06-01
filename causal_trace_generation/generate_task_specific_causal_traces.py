


from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azure_openai_client import initialize_openai_client, call_model_api, build_request_payload_input
from causal_trace_prompts import (
    ALL_TASK_NAMES,
    TASK_FAMILY_BY_NAME,
    TRACE_PIPELINE_ID,
    build_causal_trace_user_prompt,
    get_system_prompt,
    get_task_trace_contract,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_causal_traces")






class CoTLlm:


    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        max_retries: int = 3,
        retry_backoff: float = 5.0,
        timeout_sec: float = 180.0,
        reasoning_effort: str = "high",
    ):
        self.client = initialize_openai_client()
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout_sec = timeout_sec
        self.reasoning_effort = reasoning_effort


        self._lock = threading.Lock()
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def call(
        self,
        *,
        system_prompt: str,
        user_text: str = "",
        user_content: Any = None,
        max_tokens: int = 2048,
    ) -> Tuple[str, Dict[str, int]]:

        content = user_content if user_content is not None else user_text
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        payload = build_request_payload_input(messages)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                text, usage = call_model_api(
                    self.client,
                    model_name=self.model,
                    payload_input=payload,
                    timeout_sec=self.timeout_sec,
                    max_tokens=max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                with self._lock:
                    self.total_api_calls += 1
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                return text.strip(), usage
            except Exception as e:
                last_err = e
                err_str = str(e).lower()

                is_rate_limit = ("429" in err_str or "rate" in err_str
                                 or "too many" in err_str or "ratelimit" in err_str)
                if is_rate_limit:

                    base = min(15.0 * (3 ** attempt), 120.0)
                    jitter = base * random.uniform(-0.25, 0.25)
                    wait = base + jitter
                    logger.warning(f"Rate limit (attempt {attempt+1}/{self.max_retries}): "
                                   f"{e}. Backing off {wait:.1f}s...")
                else:

                    wait = self.retry_backoff * (attempt + 1)
                    logger.warning(f"API call attempt {attempt+1}/{self.max_retries} failed: {e}. "
                                   f"Retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")

    def token_summary(self) -> str:
        with self._lock:
            return (f"api_calls={self.total_api_calls} "
                    f"prompt_tokens={self.total_prompt_tokens} "
                    f"completion_tokens={self.total_completion_tokens} "
                    f"total_tokens={self.total_prompt_tokens + self.total_completion_tokens}")






class PlanCache:


    def __init__(self):
        self._cache: Dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()

    def get(self, source_path: str) -> Optional[dict]:
        with self._lock:
            if source_path in self._cache:
                return self._cache[source_path]

        plan = self._load(source_path)

        with self._lock:
            self._cache[source_path] = plan
        return plan

    @staticmethod
    def _load(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def find_step(self, plan: dict, step_id: Optional[int] = None,
                  step_goal: Optional[str] = None) -> Optional[dict]:

        steps = plan.get("steps", [])
        if step_id is not None:
            for s in steps:
                if s.get("step_id") == step_id:
                    return s
        if step_goal:
            goal_lower = step_goal.lower()[:60]
            for s in steps:
                sg = s.get("step_goal", "")
                if isinstance(sg, str) and goal_lower in sg.lower():
                    return s
        return steps[0] if steps else None









_TASK_MIN_CHARS = {
    "Task_08_Goal_Recognition": 1000,
    "Task_09_Macro_Anchor_Extraction": 950,
    "Task_10_Clip_to_Step_Goal": 1000,
    "Task_11_Action_Phrase": 1050,
    "Task_04_Affordance_Visual_Semantics": 1000,
    "Task_12_State_Evolution": 1050,
    "Task_05_Holistic_Causal_Chain": 1200,
    "Task_13_Strategic_Rationale": 1100,
    "Task_01_Spatial_Precondition": 950,
    "Task_02_Affordance_Precondition": 1150,
    "Task_03_Physical_Feasibility": 1250,
    "Task_06_Spatial_Postcondition": 950,
    "Task_07_Affordance_Postcondition": 1000,
    "Task_14_Inter_Step_Dependency": 1100,
    "Task_15_Next_Step_Prediction": 1250,
    "Task_16_Middle_Steps_Infill": 1250,
    "Task_17_Next_K_Steps_Prediction": 1200,
    "Task_18_Bad_Plan_Diagnosis_And_Repair": 1250,
    "Task_19_Counterfactual_Outcome": 1400,
    "Task_20_Failure_Recovery": 1300,
}




_SHORT_ANSWER_TASKS = {
    "Task_11_Action_Phrase",
    "Task_03_Physical_Feasibility",
    "Task_08_Goal_Recognition",
    "Task_10_Clip_to_Step_Goal",
    "Task_19_Counterfactual_Outcome",
    "Task_15_Next_Step_Prediction",
    "Task_09_Macro_Anchor_Extraction",
    "Task_02_Affordance_Precondition",
    "Task_01_Spatial_Precondition",
}

_STRUCTURE_SENSITIVE_RETRY_TASKS = {
    "Task_05_Holistic_Causal_Chain",
}



_TASK_INDICATOR_DENSITY = {
    "Task_05_Holistic_Causal_Chain": 0.03,
    "Task_03_Physical_Feasibility": 0.04,
    "Task_19_Counterfactual_Outcome": 0.04,
    "Task_18_Bad_Plan_Diagnosis_And_Repair": 0.025,
    "Task_13_Strategic_Rationale": 0.025,
    "Task_14_Inter_Step_Dependency": 0.035,
    "Task_12_State_Evolution": 0.02,
    "Task_01_Spatial_Precondition": 0.02,
    "Task_02_Affordance_Precondition": 0.02,
    "Task_06_Spatial_Postcondition": 0.02,
    "Task_07_Affordance_Postcondition": 0.02,
    "Task_11_Action_Phrase": 0.025,
    "Task_04_Affordance_Visual_Semantics": 0.025,
    "Task_16_Middle_Steps_Infill": 0.02,
    "Task_09_Macro_Anchor_Extraction": 0.025,
    "Task_15_Next_Step_Prediction": 0.025,
    "Task_17_Next_K_Steps_Prediction": 0.025,
}



_REASONING_INDICATORS = {
    "because", "therefore", "since", "implies", "indicates", "suggests",
    "confirms", "enables", "prevents", "requires", "depends", "causes",
    "leads", "results", "means", "consequently", "however", "although",
    "whereas", "instead", "otherwise", "without", "if", "would",
    "could", "cannot", "insufficient", "necessary", "sufficient",
    "gravity", "friction", "force", "torque", "leverage", "pressure",
    "trajectory", "contact", "surface", "grip", "rigid", "flexible",
    "weight", "momentum", "constraint", "mechanism", "precondition",
    "postcondition", "affordance", "state", "spatial", "functional",
    "counterfactual", "failure", "recovery", "restore", "cascade",
    "downstream", "upstream", "dependency", "condition", "effect",
    "transition", "disambiguates", "alternative", "preconditions",
    "postconditions", "affordances", "states", "conditions", "effects",
    "transitions", "dependencies", "restores", "restored", "restoring",
}



_TASK_STRUCTURE_CLUSTERS = {
    "Task_05_Holistic_Causal_Chain": {
        "spatial_pre": {"position", "located", "placed", "reach", "proximity",
                        "adjacent", "within", "above", "below", "beside", "resting"},
        "affordance_pre": {"open", "closed", "graspable", "rigid", "flexible",
                           "empty", "filled", "hot", "cold", "sharp", "accessible"},
        "mechanism": {"force", "push", "pull", "rotate", "lift", "press",
                      "torque", "grip", "contact", "motion", "slide", "friction"},
        "spatial_post": {"move", "shift", "transfer", "displace", "repositioned",
                         "new position", "now located", "relocated"},
        "affordance_post": {"now open", "now closed", "enabled", "accessible",
                            "available", "changed", "state change", "transformed"},
    },
    "Task_03_Physical_Feasibility": {
        "spatial_check": {"position", "located", "reach", "placement",
                          "distance", "proximity", "aligned", "oriented"},
        "affordance_check": {"open", "closed", "functional", "state",
                             "rigid", "flexible", "empty", "filled", "available"},
        "feasibility": {"feasib", "compatible", "sufficient", "combine",
                        "joint", "simultaneously", "together", "both"},
        "verdict": {"therefore", "verdict", "conclude", "feasible",
                    "infeasible", "verified", "confirmed", "satisfied"},
    },
    "Task_19_Counterfactual_Outcome": {
        "normal_chain": {"normal chain", "undisrupted", "baseline", "typically",
                         "precondition", "enables", "produces", "postcondition",
                         "without the disruption", "would succeed", "normally"},
        "disruption": {"disruption", "disrupted", "counterfactual", "changed",
                       "violat", "severs", "breaks the", "altered",
                       "hypothetical", "what if", "disruption point"},
        "cascade": {"immediate", "hop 1", "hop 2", "hop 3", "downstream",
                    "secondary", "tertiary", "consequence", "propagat",
                    "cascade", "furthest", "because of that",
                    "now unmet", "first thing that goes wrong"},
        "mechanism": {"force", "friction", "gravity", "contact", "surface",
                      "grip", "rigid", "momentum", "pressure", "torque",
                      "constraint", "mechanism", "leverage", "shear",
                      "physical principle"},
    },
}


def _final_sentence_overlap(cot_text: str, answer: str,
                             n_tail_sentences: int = 2) -> float:

    raw_sents = re.split(r'(?<=[.!?])\s+', cot_text.strip())
    sentences = [s.strip() for s in raw_sents if len(s.strip()) > 10]

    if len(sentences) < 3:

        return 0.0

    def _rough_stem(word: str) -> str:

        w = word.lower().strip(".,;:!?()[]\"'")


        for suffix in ("tion", "sion", "ment", "ness", "ally",
                       "ying", "ing",
                       "ous", "ive", "ful", "less", "able", "ible",
                       "ied", "ies", "ely", "ed", "ly", "es", "er", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    def content_words(text: str) -> list:
        return [_rough_stem(w) for w in text.split()
                if len(w.strip(".,;:!?()[]\"'")) > 2]

    answer_stems = set(content_words(answer))

    if not answer_stems:
        return 0.0

    def _overlap_for_tail(tail_text: str) -> float:
        tw = content_words(tail_text)
        if not tw:
            return 0.0
        matches = sum(1 for w in tw if w in answer_stems)
        return matches / len(tw)


    tail_1 = sentences[-1]
    tail_2 = " ".join(sentences[-n_tail_sentences:])
    return max(_overlap_for_tail(tail_1), _overlap_for_tail(tail_2))


def _compute_novelty_ratio(cot_text: str, answer: str) -> float:

    def ngrams(text: str, n: int = 4) -> set:
        words = text.lower().split()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    cot_ng = ngrams(cot_text)
    ans_ng = ngrams(answer)
    if not cot_ng:
        return 0.0
    overlap = len(cot_ng & ans_ng)
    return 1.0 - (overlap / len(cot_ng))


def _compute_self_repetition(cot_text: str) -> float:


    words = cot_text.lower().split()
    half_score = 0.0
    if len(words) >= 20:
        mid = len(words) // 2
        def ngrams_4(word_list):
            return {tuple(word_list[i:i+4]) for i in range(len(word_list) - 3)}
        first_ng = ngrams_4(words[:mid])
        second_ng = ngrams_4(words[mid:])
        if second_ng:
            overlap = len(first_ng & second_ng)
            half_score = overlap / len(second_ng)


    import re
    raw_sents = re.split(r'(?<=[.!?])\s+', cot_text.strip())
    sentences = [s.strip() for s in raw_sents if len(s.strip()) > 15]
    pair_score = 0.0
    if len(sentences) >= 4:
        def trigrams(text: str) -> set:
            w = text.lower().split()
            return {tuple(w[i:i+3]) for i in range(len(w) - 2)}
        sent_tg = [trigrams(s) for s in sentences]
        rep_pairs = 0
        total_pairs = 0
        for i in range(len(sent_tg)):
            for j in range(i + 2, len(sent_tg)):                 
                if not sent_tg[i] or not sent_tg[j]:
                    continue
                total_pairs += 1
                intersection = len(sent_tg[i] & sent_tg[j])
                union = len(sent_tg[i] | sent_tg[j])
                if union > 0 and intersection / union > 0.3:               
                    rep_pairs += 1
        pair_score = rep_pairs / total_pairs if total_pairs > 0 else 0.0

    return max(half_score, pair_score)


def _count_template_patterns(cot_text: str) -> Tuple[int, List[str]]:

    patterns = []



    caps_colon = re.findall(r'^([A-Z][A-Z\s]{2,30}):', cot_text, re.MULTILINE)
    for match in caps_colon:
        patterns.append(f"CAPS_COLON: {match.strip()}")


    para_n = re.findall(r'PARAGRAPH\s*\d', cot_text)
    for match in para_n:
        patterns.append(f"PARA_NUM: {match}")



    numbered = re.findall(r'^\d+\.\s', cot_text, re.MULTILINE)
    if len(numbered) >= 5:
        patterns.append(f"NUMBERED_LIST: {len(numbered)} items")


    step_heading = re.findall(
        r'^(?:Step|Phase|Layer|Stage|Part)\s*\d+\s*[:\-\u2014]',
        cot_text, re.MULTILINE | re.IGNORECASE
    )
    for match in step_heading:
        patterns.append(f"STEP_HEADING: {match.strip()}")

    return len(patterns), patterns




_HEADING_ARTIFACT_RE = re.compile(
    r'^(?:'
    r'(?:PARAGRAPH\s*\d+\s*[\u2014\u2013\-]\s*)'                      
    r'|(?:[A-Z][A-Z\s]{2,30}:\s*)'                                    
    r'|(?:[A-Z][A-Z\s]{2,30}[\u2014\u2013\-]\s*)'                   
    r')',
    re.MULTILINE,
)


def _strip_heading_artifacts(cot_text: str) -> str:

    cleaned = cot_text
    for _ in range(3):
        next_pass = _HEADING_ARTIFACT_RE.sub('', cleaned)
        if next_pass == cleaned:
            break
        cleaned = next_pass

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _check_task_structure(cot_text: str, task_name: str) -> Tuple[bool, str]:

    clusters = _TASK_STRUCTURE_CLUSTERS.get(task_name)
    if not clusters:
        return True, "ok"

    lower = cot_text.lower()
    present = 0
    missing_names = []
    for name, keywords in clusters.items():
        if any(kw in lower for kw in keywords):
            present += 1
        else:
            missing_names.append(name)

    required = max(1, int(len(clusters) * 0.6))                         
    if present < required:
        return False, (f"Missing structural layers for {task_name}: "
                       f"{', '.join(missing_names)} ({present}/{len(clusters)} present, "
                       f"need >= {required})")
    return True, "ok"


def _check_causal_trace_contract(trace_text: str, task_name: str) -> Tuple[bool, str]:

    contract = get_task_trace_contract(task_name)
    clusters = contract.get("validation_clusters") or []
    if not clusters:
        return True, "ok"

    lower = trace_text.lower()
    present = []
    missing = []
    for cluster in clusters:
        name = cluster.get("name", "unknown")
        keywords = cluster.get("keywords", [])
        if any(str(keyword).lower() in lower for keyword in keywords):
            present.append(name)
        else:
            missing.append(name)

    required = max(3, int(len(clusters) * 0.6))
    if len(present) < required:
        return False, (
            f"Causal trace misses task contract coverage for {task_name}: "
            f"{', '.join(missing)} ({len(present)}/{len(clusters)} present, need >= {required})"
        )



    task_cluster = clusters[-1].get("name") if clusters else ""
    if task_cluster and task_cluster not in present:
        return False, (
            f"Causal trace misses task-specific contract cluster: {task_cluster}"
        )

    return True, "ok"


def _check_task_hard_requirements(trace_text: str, task_name: str) -> Tuple[bool, str]:

    lower = trace_text.lower()

    if task_name == "Task_02_Affordance_Precondition":
        property_terms = [
            "open", "unsealed", "rigid", "flexible", "empty", "intact",
            "grasp", "accessible", "obstruct", "deform", "solid",
            "closed", "sealed", "sharp", "stable", "supported", "reachable",
            "unobstructed", "movable", "visible", "available", "firm",
            "loose", "taut", "flat", "dry", "clean", "handle", "edge",
        ]
        property_hits = sum(1 for term in property_terms if term in lower)
        provenance_hits = any(
            term in lower for term in ["earlier", "already", "before", "prior", "created", "maintained"]
        )
        failure_hits = sum(
            1 for term in ["if", "without", "would fail", "could not", "cannot",
                           "would not", "blocked", "jam", "collapse", "stuck",
                           "slip", "spill", "break", "prevents"]
            if term in lower
        )
        dependency_hits = any(
            term in lower for term in ["force path", "contact path", "sub-motion",
                                       "motion", "force", "depends", "contact",
                                       "grip", "friction", "leverage", "support",
                                       "containment", "trajectory", "pressure"]
        )
        if property_hits < 3:
            return False, "Task_10 trace must discuss at least three concrete affordance properties"
        if not provenance_hits:
            return False, "Task_10 trace must state prerequisite provenance (already/before/earlier/created/maintained)"
        if failure_hits < 2:
            return False, "Task_10 trace must include concrete failure tests for missing affordance properties"
        if not dependency_hits:
            return False, "Task_10 trace must tie properties to a force/contact/sub-motion dependency"

    if task_name == "Task_15_Next_Step_Prediction":
        has_state_inventory = any(term in lower for term in ["prefix", "current", "established state", "cumulative state"])
        has_remaining = any(term in lower for term in ["remaining", "still", "unmet", "not yet"])
        has_alternative = any(term in lower for term in ["alternative", "candidate", "tempting", "weaker", "rather than", "instead"])
        has_earliest_now = any(term in lower for term in ["earliest", "successor", "now", "immediately"])
        has_downstream = any(term in lower for term in ["downstream", "later", "following", "next step", "postcondition"])
        if not has_state_inventory:
            return False, "Task_15 trace must inventory the prefix/current cumulative state"
        if not has_remaining:
            return False, "Task_15 trace must identify remaining or unmet goal requirements"
        if not has_alternative:
            return False, "Task_15 trace must compare an alternative/candidate next action"
        if not has_earliest_now:
            return False, "Task_15 trace must justify why the step is the earliest valid successor now"
        if not has_downstream:
            return False, "Task_15 trace must connect the predicted step to a downstream/later postcondition"

    return True, "ok"


def validate_cot(cot_text: str, answer: str, task_name: str) -> Tuple[bool, str]:

    if not cot_text or not cot_text.strip():
        return False, "Causal trace is empty"

    cot_stripped = cot_text.strip()
    cot_len = len(cot_stripped)


    min_chars = _TASK_MIN_CHARS.get(task_name, 200)


    answer_len = len(answer.strip())
    if task_name in _SHORT_ANSWER_TASKS:
        adaptive_min = max(min_chars, int(answer_len * 4.0))
    else:
        adaptive_min = max(min_chars, int(answer_len * 2.5))

    adaptive_min = min(adaptive_min, 1800)

    if cot_len < adaptive_min:
        return False, (f"Causal trace too short ({cot_len} chars, need >= {adaptive_min} "
                       f"[task_min={min_chars}, 2x_answer={int(answer_len*2.0)}])")


    sentences = [s.strip() for s in cot_stripped.replace("\n", ". ").split(". ")
                 if len(s.strip()) > 10]
    min_sentences = 5 if task_name in _TASK_MIN_CHARS else 3
    if len(sentences) < min_sentences:
        return False, (f"Causal trace has too few reasoning steps ({len(sentences)}, "
                       f"need >= {min_sentences})")




    _KEYWORD_OVERLAP_THRESHOLD = {
        "Task_09_Macro_Anchor_Extraction": 0.10,
        "Task_18_Bad_Plan_Diagnosis_And_Repair": 0.10,
    }
    kw_threshold = _KEYWORD_OVERLAP_THRESHOLD.get(task_name, 0.15)
    answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
    cot_words = set(w.lower() for w in cot_stripped.split() if len(w) > 3)
    if answer_words:
        overlap = len(answer_words & cot_words) / len(answer_words)
        if overlap < kw_threshold:
            return False, (f"Causal trace has low keyword overlap with answer "
                           f"({overlap:.1%}, need >= {kw_threshold:.0%})")


    if answer.strip() in cot_stripped:
        return False, "Causal trace contains verbatim copy of the answer"


    novelty = _compute_novelty_ratio(cot_stripped, answer)
    if novelty < 0.50:
        return False, (f"Causal trace is mostly restatement of the answer "
                       f"(novelty={novelty:.1%}, need >= 50%)")


    cot_words_lower = [w.lower().strip(".,;:!?()[]") for w in cot_stripped.split()]
    total_words = len(cot_words_lower)
    if total_words > 0:
        indicator_count = sum(1 for w in cot_words_lower if w in _REASONING_INDICATORS)
        indicator_density = indicator_count / total_words
        threshold = _TASK_INDICATOR_DENSITY.get(task_name, 0.03)
        if indicator_density < threshold:
            return False, (f"Causal trace lacks reasoning language "
                           f"(indicator_density={indicator_density:.1%}, "
                           f"need >= {threshold:.1%} for {task_name})")


    lower_cot = cot_stripped.lower()
    cot_start = lower_cot[:80]                                        
    cot_end = lower_cot[-100:]                                          


    always_bad = [
        ("as an ai", "AI self-reference"),
        ("i don't have access", "refusal language"),
        ("i can't see", "refusal language"),
        ("from the json", "meta-reference to data format"),
        ("the context says", "meta-reference to context"),
        ("according to the evidence provided", "meta-reference to context"),
        ("based on the provided context", "meta-reference to context"),
        ("based on the provided information", "meta-reference to context"),

        ("the user wants me to", "internal monologue leakage"),
        ("the user is asking", "internal monologue leakage"),
        ("i should think about", "internal monologue leakage"),
        ("i need to figure out", "internal monologue leakage"),
        ("my task is to", "internal monologue leakage"),
        ("the question asks me", "internal monologue leakage"),
        ("i'm being asked to", "internal monologue leakage"),
        ("let me consider", "internal monologue leakage"),
    ]
    for pattern, reason in always_bad:
        if pattern in lower_cot:
            return False, f"Causal trace contains anti-pattern: '{pattern}' ({reason})"


    start_bad = [
        ("let me think", "meta-commentary preamble"),
        ("let me analyze", "meta-commentary preamble"),
        ("let me start", "meta-commentary preamble"),
        ("i need to analyze", "meta-commentary preamble"),
        ("i need to think", "meta-commentary preamble"),
    ]
    for pattern, reason in start_bad:
        if pattern in cot_start:
            return False, f"Causal trace contains anti-pattern at start: '{pattern}' ({reason})"


    end_bad = [
        ("therefore the answer", "premature conclusion at end"),
        ("in conclusion,", "summary-style ending"),
        ("in summary,", "summary-style ending"),
    ]
    for pattern, reason in end_bad:
        if pattern in cot_end:
            return False, f"Causal trace contains anti-pattern at end: '{pattern}' ({reason})"


    rep_score = _compute_self_repetition(cot_stripped)
    if rep_score > 0.15:
        return False, (f"Causal trace is internally repetitive "
                       f"(self_repetition={rep_score:.1%}, need <= 15%)")


    struct_ok, struct_reason = _check_task_structure(cot_stripped, task_name)
    if not struct_ok:
        return False, struct_reason


    contract_ok, contract_reason = _check_causal_trace_contract(cot_stripped, task_name)
    if not contract_ok:
        return False, contract_reason

    hard_ok, hard_reason = _check_task_hard_requirements(cot_stripped, task_name)
    if not hard_ok:
        return False, hard_reason






    tail_overlap = _final_sentence_overlap(cot_stripped, answer)
    if tail_overlap > 0.80:
        return False, (f"Causal trace final sentences are near-verbatim copy of answer "
                       f"(tail_overlap={tail_overlap:.1%}, need <= 80%)")

    return True, "ok"






def load_resume_state(path: str) -> Set[str]:

    done = set()
    if not os.path.isfile(path):
        return done
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("id", "")
                    status = rec.get("status", "")
                    if status == "done" and qid:
                        done.add(qid)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return done


def _conversation_value(record: dict, index: int) -> str:
    convs = record.get("conversations", [])
    if not isinstance(convs, list) or len(convs) <= index:
        return ""
    item = convs[index]
    if not isinstance(item, dict):
        return ""
    return str(item.get("value", ""))


def _strip_trace_from_answer(value: str) -> str:

    if "</think>" in value:
        return value.split("</think>", 1)[1].strip()
    return value.strip()


def _qa_instance_fingerprint(record: dict) -> str:

    meta = record.get("meta", {})
    if isinstance(meta, dict):
        meta_core = {k: v for k, v in meta.items() if k != "causal_trace"}
    else:
        meta_core = {}
    payload = {
        "id": record.get("id", ""),
        "question": _conversation_value(record, 0).strip(),
        "answer": _strip_trace_from_answer(_conversation_value(record, 1)),
        "image": record.get("image", None),
        "video": record.get("video", None),
        "meta": meta_core,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_output_fingerprints(path: str) -> Dict[str, int]:

    counts: Dict[str, int] = defaultdict(int)
    if not os.path.isfile(path):
        return counts
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts[_qa_instance_fingerprint(rec)] += 1
    except Exception:
        pass
    return counts


def append_jsonl(path: str, record: dict):

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")






class IssueLogger:


    def __init__(self, output_dir: str):
        self.errors_path = os.path.join(output_dir, "issues_errors.jsonl")
        self.warnings_path = os.path.join(output_dir, "issues_warnings.jsonl")
        self._lock = threading.Lock()
        self.error_count = 0
        self.warning_count = 0

    def log(self, severity: str, message: str, **extra):
        rec = {"severity": severity, "message": message, **extra}
        with self._lock:
            if severity == "error":
                self.error_count += 1
                append_jsonl(self.errors_path, rec)
            else:
                self.warning_count += 1
                append_jsonl(self.warnings_path, rec)






_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_VISUAL_IO_SEMAPHORE: Optional[threading.BoundedSemaphore] = None


class _VisualIoSlot:


    def __enter__(self):
        if _VISUAL_IO_SEMAPHORE is not None:
            _VISUAL_IO_SEMAPHORE.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _VISUAL_IO_SEMAPHORE is not None:
            _VISUAL_IO_SEMAPHORE.release()
        return False


def _as_path_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _safe_isfile(path: str, retries: int = 4) -> bool:

    for attempt in range(retries):
        try:
            if os.path.isfile(path):
                return True
        except OSError:
            pass
        if attempt + 1 < retries:
            time.sleep(0.35 * (attempt + 1))
    return False


def _split_local_evidence_paths(qa_record: dict) -> Tuple[List[str], List[str]]:

    meta = qa_record.get("meta") if isinstance(qa_record.get("meta"), dict) else {}
    raw_paths = []
    raw_paths.extend(_as_path_list(qa_record.get("image")))
    raw_paths.extend(_as_path_list(qa_record.get("video")))
    raw_paths.extend(_as_path_list(meta.get("evidence_files")))

    images: List[str] = []
    videos: List[str] = []
    for path in raw_paths:
        if not _safe_isfile(path):
            continue
        ext = Path(path).suffix.lower()
        if ext in _IMAGE_EXTS and path not in images:
            images.append(path)
        elif ext in _VIDEO_EXTS and path not in videos:
            videos.append(path)
    return images, videos


def _video_duration_sec(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    for attempt in range(5):
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            duration = float(proc.stdout.strip())
            return duration if duration > 0 else None
        except Exception:
            if attempt < 4:
                time.sleep(0.75 * (attempt + 1))
    return None


def _run_ffmpeg_frame_extract(commands: List[List[str]], out_path: str) -> bool:

    for cmd in commands:
        for attempt in range(3):
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                    return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.75 * (attempt + 1))
    return False


def _sample_video_frames(video_path: str, temp_dir: str, max_frames: int) -> List[str]:

    if max_frames <= 0:
        return []
    duration = _video_duration_sec(video_path)
    if duration:
        if max_frames == 1:
            timestamps = [duration * 0.5]
        else:
            fractions = [0.12, 0.35, 0.65, 0.88][:max_frames]
            timestamps = [max(0.05, min(duration - 0.05, duration * frac)) for frac in fractions]
    else:
        timestamps = [0.0]

    outputs: List[str] = []
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_path).stem)[:80]
    local_video_path: Optional[str] = None

    def _commands_for(src_video: str, ts_value: float, dst_path: str) -> List[List[str]]:
        return [
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts_value:.3f}", "-i", src_video,
                "-frames:v", "1", "-vf", "scale='min(768,iw)':-2",
                "-q:v", "3", "-y", dst_path,
            ],
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", src_video, "-ss", f"{ts_value:.3f}",
                "-frames:v", "1", "-vf", "scale='min(768,iw)':-2",
                "-q:v", "3", "-y", dst_path,
            ],
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", src_video, "-vf", "thumbnail,scale='min(768,iw)':-2",
                "-frames:v", "1", "-q:v", "3", "-y", dst_path,
            ],
        ]

    for idx, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(temp_dir, f"{stem}_sample_{idx:02d}.jpg")
        if _run_ffmpeg_frame_extract(_commands_for(video_path, ts, out_path), out_path):
            outputs.append(out_path)
            continue

        if local_video_path is None:
            local_video_path = os.path.join(temp_dir, f"{stem}_local_copy{Path(video_path).suffix}")
            copy_error: Optional[Exception] = None
            for copy_attempt in range(3):
                try:
                    shutil.copy2(video_path, local_video_path)
                    copy_error = None
                    break
                except Exception as exc:
                    copy_error = exc
                    time.sleep(0.75 * (copy_attempt + 1))
            if copy_error is not None:
                logger.warning(f"Could not copy video evidence locally {video_path}: {copy_error}")
                local_video_path = ""

        if local_video_path and _run_ffmpeg_frame_extract(_commands_for(local_video_path, ts, out_path), out_path):
            outputs.append(out_path)
        else:
            logger.warning(f"Could not sample video evidence {video_path} at {ts:.2f}s")
    return outputs


def build_user_content_with_evidence(
    *,
    user_prompt: str,
    qa_record: dict,
    evidence_mode: str = "auto",
    max_images: int = 4,
    max_video_frames: int = 4,
) -> Tuple[Any, List[str], Optional[str], Dict[str, int]]:

    if evidence_mode not in {"text_only", "auto", "require_readable"}:
        raise ValueError(f"Unknown evidence_mode: {evidence_mode}")
    if evidence_mode == "text_only":
        return user_prompt, [], None, {
            "candidate_image_paths": 0,
            "candidate_video_paths": 0,
            "sampled_video_frames": 0,
            "attached_visuals": 0,
        }

    warnings: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix="causal_trace_video_frames_")
    with _VisualIoSlot():
        image_paths, video_paths = _split_local_evidence_paths(qa_record)

        sampled_frames: List[str] = []
        for video_path in video_paths:
            sampled_frames.extend(_sample_video_frames(video_path, temp_dir, max_video_frames))

    visual_paths = image_paths[:max_images] + sampled_frames[:max_video_frames]
    evidence_stats = {
        "candidate_image_paths": len(image_paths),
        "candidate_video_paths": len(video_paths),
        "sampled_video_frames": len(sampled_frames),
        "attached_visuals": len(visual_paths),
    }
    if not visual_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if evidence_mode == "require_readable":
            raise FileNotFoundError("No readable image evidence or sampled video frames were available.")
        warnings.append("No readable visual evidence attached; using text-only prompt.")
        return user_prompt, warnings, None, evidence_stats

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                user_prompt
                + "\n\n=== VISUAL GROUNDING INSTRUCTION ===\n"
                "Use the attached visual evidence only to verify object identity, layout, contact, and motion cues. "
                "Do not mention image files, frames, sampling, filenames, or paths in the trace."
            ),
        }
    ]
    for idx, path in enumerate(visual_paths, start=1):
        content.append({"type": "text", "text": f"Visual evidence {idx}"})
        content.append({"type": "image_url", "image_url": {"url": path}})

    return content, warnings, temp_dir, evidence_stats













_RETRY_SCAFFOLDS = {
    "Task_09_Macro_Anchor_Extraction": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, introduce the exact anchor terms from the answer in causal "
        "sentences, not as a copied list. For every selected anchor, name "
        "whether it functions as a tool, patient, container, support surface, "
        "target object, alignment reference, or state-bearing object.\n"
        "Second, for each anchor, explain the visible manipulation or state "
        "change that makes it causally central. If an item is touched, moved, "
        "opened, closed, held, filled, aligned, supported, or used as a tool, "
        "state that mechanism directly.\n"
        "Third, run the removal test for every anchor: if that anchor were "
        "absent or in the wrong state, what later motion, contact, containment, "
        "alignment, or state transition would fail? Contrast at least one "
        "background or bystander object that is visible but not an anchor.\n"
        "Fourth, synthesize why the selected anchors, together, carry the "
        "macro structure of the task. End with the dependency pattern, not a "
        "repeated answer list."
    ),
    "Task_01_Spatial_Precondition": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, name the required spatial position of each relevant object "
        "before the action starts. Use concrete relation words such as above, "
        "inside, beside, aligned with, facing, under, near, within reach, or "
        "clear of obstruction when supported by the evidence.\n"
        "Second, explain orientation and alignment: which surface, edge, "
        "opening, handle, tool tip, or target area must face which direction, "
        "and why that orientation creates a usable contact path or trajectory.\n"
        "Third, explain proximity and reach: why the hand, tool, or object "
        "must be close enough for contact, support, containment, leverage, or "
        "controlled transfer. Include what physical sub-motion would fail if "
        "the proximity were wrong.\n"
        "Fourth, run two spatial failure tests using explicit counterfactual "
        "language. If the position, orientation, alignment, reach, support, or "
        "trajectory were missing, state exactly how the action would be blocked."
    ),
    "Task_02_Affordance_Precondition": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, identify at least three concrete affordance properties that "
        "matter for the action. Use specific property language when supported "
        "by the evidence, such as open or closed state, graspable handle or "
        "edge, accessible surface, rigid or flexible material, empty or "
        "contained interior, stable support, unobstructed path, intact part, "
        "sharp working edge, dry or clean contact surface. Do not invent a "
        "property; if a property is not required, contrast it with the ones "
        "that are required.\n"
        "Second, state provenance for each prerequisite: already true before "
        "this clip, created by an earlier step, maintained by the current "
        "setup, or directly visible now. Tie that provenance to the evidence "
        "instead of making a generic checklist.\n"
        "Third, run at least two missing-property failure tests. For each, "
        "use explicit counterfactual language such as if, without, could not, "
        "would fail, blocked, jammed, slipped, spilled, collapsed, or stuck, "
        "and describe the physical sub-motion that would fail.\n"
        "Fourth, connect the properties to the force or contact dependency: "
        "where the hand or tool applies force, what grip, friction, support, "
        "containment, pressure, leverage, contact path, or trajectory carries "
        "the action, and why all prerequisites jointly make the action ready."
    ),
    "Task_11_Action_Phrase": (
        "Write your thinking in five distinct paragraphs, each at least "
        "2-3 sentences. Cover these aspects in order:\n"
        "First, describe the raw motion you observe — what body part or "
        "tool moves, in what trajectory, at what speed, what surfaces "
        "are in contact.\n"
        "Second, name 2-3 candidate action verbs and for each state what "
        "observable feature would confirm it. Then identify which feature "
        "is present and which verb it selects.\n"
        "Third, explain how the patient object responds to the force — "
        "does it deform, translate, rotate, change state? What does this "
        "response confirm about the action type?\n"
        "Fourth, explain why this particular tool or hand configuration "
        "is suited to this action and what would go wrong with a "
        "different approach.\n"
        "Fifth, show how the kinematic observation, verb selection, and "
        "object response converge on the final action phrase."
    ),
    "Task_05_Holistic_Causal_Chain": (
        "Write your thinking in five distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, state the spatial preconditions before the action: where the "
        "hand, tool, target object, support surface, and relevant container or "
        "obstacle are located, whether they are within reach, and how their "
        "position or proximity enables contact.\n"
        "Second, state the affordance preconditions before the action: whether "
        "the relevant object is open or closed, graspable or inaccessible, "
        "empty or filled, rigid or flexible, hot or cold, sharp or dull, and "
        "which functional state matters for the action.\n"
        "Third, explain the physical mechanism that links the preconditions to "
        "the action: name the force, push, pull, lift, rotate, press, grip, "
        "contact, slide, friction, torque, or motion path that makes the "
        "transition possible.\n"
        "Fourth, describe the spatial postconditions: what moved, shifted, "
        "transferred, displaced, or was repositioned, and what new position or "
        "location exists after the action.\n"
        "Fifth, describe the affordance postconditions: what state changed, "
        "what is now open, closed, enabled, accessible, available, transformed, "
        "or otherwise ready for a downstream step. Connect the final state "
        "back to the answer instead of merely repeating it."
    ),
    "Task_03_Physical_Feasibility": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, for each spatial precondition, state it, verify it "
        "against what you observe, and explain what physical motion "
        "would fail if this condition were violated.\n"
        "Second, for each affordance precondition, state it, verify the "
        "object's current functional state, and explain the physical "
        "dependency.\n"
        "Third, explain how spatial and affordance conditions combine — "
        "are they mutually compatible? Does any spatial arrangement "
        "conflict with an affordance requirement?\n"
        "Fourth, derive the yes/no conclusion explicitly from the checks "
        "above, referencing which specific verifications support it."
    ),
    "Task_19_Counterfactual_Outcome": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, describe the undisrupted causal chain — name the objects, "
        "their spatial arrangement, the physical mechanism of the action, "
        "and the postconditions it normally produces.\n"
        "Second, identify which link in the chain breaks and explain WHY "
        "at a physical level — name the principle that is violated (friction, "
        "gravity, containment, leverage, etc.).\n"
        "Third, trace the cascade through three explicit hops: the immediate "
        "failure, the secondary condition now unmet because of that failure, "
        "and the furthest-reaching consequence for the overall plan. Each hop "
        "must be a separate inference step with physical specificity.\n"
        "Fourth, explicitly contrast the normal end-state versus the broken "
        "end-state — what is the gap between them?"
    ),
    "Task_20_Failure_Recovery": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, describe the failure state with full physical detail — name "
        "each object's current position, orientation, and contact relationships. "
        "What exactly went wrong, and where is everything now?\n"
        "Second, diagnose the root cause through physical mechanism — what force "
        "was applied, through what path, and why did it produce the failure? "
        "Name the material property or geometric constraint involved.\n"
        "Third, design recovery actions with quantitative specificity — for each "
        "sub-step, name the object, the motion direction and approximate "
        "magnitude, the target state, and why that motion restores the needed "
        "condition.\n"
        "Fourth, verify resumability — walk through each precondition of the "
        "interrupted step and confirm it is now satisfied after recovery."
    ),
    "Task_16_Middle_Steps_Infill": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, analyze the postconditions of the first step — what state "
        "exists after it completes? Name specific objects, their positions, "
        "and their functional states.\n"
        "Second, analyze the preconditions of the last step — what must be "
        "true for it to begin? Identify the gap between the first step's "
        "postconditions and the last step's preconditions.\n"
        "Third, for each missing intermediate step, explain WHY it is "
        "physically necessary — what precondition does it create that did "
        "not exist before? Use causal language: because, therefore, enables.\n"
        "Fourth, verify the complete chain: does each intermediate step's "
        "postcondition satisfy the next step's precondition?"
    ),
    "Task_13_Strategic_Rationale": (
        "Write your thinking in four distinct paragraphs, each at least "
        "3-4 sentences. Cover these aspects in order:\n"
        "First, identify the postconditions this step produces — what new "
        "state exists after it completes that did not exist before?\n"
        "Second, trace the downstream dependency — which later step requires "
        "these postconditions as preconditions? Be specific about the "
        "physical mechanism.\n"
        "Third, run the counterfactual skip test — if this step were "
        "removed entirely, what specific physical condition would be missing "
        "and why would the downstream step fail?\n"
        "Fourth, conclude with why no alternative action could substitute "
        "for this step — what makes it uniquely necessary?"
    ),
}


def _build_retry_prompt(
    original_prompt: str,
    failure_reason: str,
    task_name: str,
    answer: str,
    retry_num: int,
) -> str:




    parts = []

    if retry_num == 0:

        parts.append(
            "MANDATORY CORRECTION — YOUR PREVIOUS OUTPUT WAS REJECTED.\n"
            f"Reason: {failure_reason}\n"
        )

        if "too short" in failure_reason:
            answer_word_count = len(answer.split())
            parts.append(
                f"The answer is only {answer_word_count} words. Your thinking "
                f"must be AT LEAST 200 words (roughly 15-20 sentences). You wrote "
                f"far too little. The short answer compresses away all the "
                f"perceptual and inferential work — YOUR job is to decompress it.\n"
            )
        elif ("restatement" in failure_reason or "novelty" in failure_reason
              or "near-verbatim" in failure_reason or "tail_overlap" in failure_reason):
            parts.append(
                "Your output was a paraphrase of the answer. This is worthless. "
                "The answer already exists — you must show the REASONING PATH:\n"
                "  - What physical observations led to the conclusion?\n"
                "  - What alternative was considered and rejected?\n"
                "  - What mechanism or principle makes the conclusion inevitable?\n"
                "  - What counterfactual would change the outcome?\n"
                "Generate content the answer DOES NOT contain.\n"
                "IMPORTANT: Do NOT end your thinking by restating or summarizing "
                "the answer. Your final sentence should be about the decisive "
                "observation or mechanism — the reader can derive the answer "
                "from your analysis without you repeating it.\n"
            )
        elif "reasoning language" in failure_reason:
            parts.append(
                "Your output read like a description, not an analysis. Use "
                "causal connectives: because, therefore, implies, enables, "
                "prevents, requires, without X then Y, the mechanism is. "
                "Every sentence should advance a causal argument.\n"
            )
        elif "keyword overlap" in failure_reason:
            answer_terms = ", ".join(
                w.strip(".,;:()[]{}").lower()
                for w in answer.split()
                if len(w.strip(".,;:()[]{}")) > 3
            )
            parts.append(
                "Your output was analytically under-grounded in the answer. "
                "Do not paste the answer as a list, but every substantive "
                "answer term must appear inside a causal sentence that explains "
                "its role, state change, or removal-test failure. Required "
                f"answer terms to ground explicitly: {answer_terms}.\n"
            )
        elif "spatial_precondition" in failure_reason:
            parts.append(
                "Your output missed the spatial-precondition contract. Use "
                "explicit spatial reasoning words in natural prose: spatial "
                "position, orientation, alignment, proximity, reach, contact "
                "path, trajectory, support, and obstruction. Each relation must "
                "be tied to the physical sub-motion it enables or blocks.\n"
            )
        elif "anti-pattern" in failure_reason:
            parts.append(
                "Start your FIRST WORD with a concrete physical observation "
                "(an object name, a body part, a spatial relationship). "
                "No preambles. No meta-commentary. No 'Let me' or 'I need to'.\n"
            )


        scaffold = _RETRY_SCAFFOLDS.get(task_name)
        if scaffold:
            parts.append(scaffold)
            parts.append("")

        parts.append(original_prompt)

    else:

        parts.append(
            "THIS IS YOUR FINAL ATTEMPT. Your previous two outputs were both "
            f"rejected ({failure_reason}). If you produce another short or "
            "shallow response, this record will be marked as a failure.\n\n"
            "HARD REQUIREMENT: Write at least 250 words of analytical prose. "
            "Count your sentences — you need at least 15 sentences, each "
            "advancing the reasoning. If you finish and your output looks "
            "shorter than a medium paragraph, you have failed again.\n\n"
            "STRATEGY: Before writing the conclusion, build up from "
            "observations. Start with what you physically see (objects, "
            "positions, materials, motions). Then analyze what these "
            "observations physically imply. Then consider what alternatives "
            "exist and why they don't fit. Then trace the causal chain to "
            "the conclusion. Each of these four phases should be at least "
            "3 sentences.\n"
        )

        scaffold = _RETRY_SCAFFOLDS.get(task_name)
        if scaffold:
            parts.append(scaffold)
            parts.append("")

        parts.append(original_prompt)

    return "\n".join(parts)






def process_one_qa(
    qa_record: dict,
    *,
    llm: CoTLlm,
    plan_cache: PlanCache,
    task_name: str,
    evidence_mode: str = "auto",
    max_images: int = 4,
    max_video_frames: int = 4,
) -> Dict[str, Any]:

    result = {"record": None, "status": "skipped", "error": None, "usage": {}}

    qid = qa_record.get("id", "")
    convs = qa_record.get("conversations", [])
    if len(convs) < 2:
        result["status"] = "error"
        result["error"] = "Missing conversations"
        return result

    question = convs[0].get("value", "")
    answer = convs[1].get("value", "")
    meta = qa_record.get("meta", {})
    llm_fields = meta.get("llm_fields", {})
    source_path = meta.get("source_path", "")

    if not question or not answer:
        result["status"] = "error"
        result["error"] = "Empty question or answer"
        return result


    plan_context = None
    if source_path and os.path.isfile(source_path):
        plan = plan_cache.get(source_path)
        if plan:
            plan_context = {"high_level_goal": plan.get("high_level_goal", "")}
            steps = plan.get("steps", [])
            plan_context["all_step_goals"] = [
                s.get("step_goal", "") for s in steps if isinstance(s, dict)
            ]

            step_goal = llm_fields.get("step_goal") or ""
            step_id = llm_fields.get("step_id")
            step = plan_cache.find_step(plan, step_id=step_id, step_goal=step_goal) if (step_goal or step_id) else None
            if step:
                plan_context["step"] = step
                for idx, candidate in enumerate(steps):
                    if candidate is step or candidate.get("step_id") == step.get("step_id"):
                        if idx > 0:
                            plan_context["previous_step"] = steps[idx - 1]
                        if idx + 1 < len(steps):
                            plan_context["next_step"] = steps[idx + 1]
                        break


    system_prompt = get_system_prompt(task_name)
    user_prompt = build_causal_trace_user_prompt(
        task_name=task_name,
        question=question,
        answer=answer,
        llm_fields=llm_fields,
        plan_context=plan_context,
    )

    all_evidence_warnings: List[str] = []
    attempted_evidence_stats = {
        "candidate_image_paths": 0,
        "candidate_video_paths": 0,
        "sampled_video_frames": 0,
        "attached_visuals": 0,
    }
    final_evidence_warnings: List[str] = []
    final_evidence_stats = dict(attempted_evidence_stats)

    def _merge_attempted_stats(stats: Dict[str, int]) -> None:
        for key, value in stats.items():
            attempted_evidence_stats[key] = max(attempted_evidence_stats.get(key, 0), value)

    def _call_with_prompt(prompt: str) -> Tuple[str, Dict[str, int], List[str], Dict[str, int]]:
        user_content, warnings, temp_dir, evidence_stats = build_user_content_with_evidence(
            user_prompt=prompt,
            qa_record=qa_record,
            evidence_mode=evidence_mode,
            max_images=max_images,
            max_video_frames=max_video_frames,
        )
        all_evidence_warnings.extend(warnings)
        _merge_attempted_stats(evidence_stats)
        try:
            text, usage = llm.call(
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=4096,
            )
            return text, usage, warnings, evidence_stats
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


    try:
        cot_text, usage, final_evidence_warnings, final_evidence_stats = _call_with_prompt(user_prompt)
        result["usage"] = usage
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"LLM call failed: {e}"
        return result


    is_valid, reason = validate_cot(cot_text, answer, task_name)
    max_retries = 2 if (
        task_name in _SHORT_ANSWER_TASKS
        or task_name in _STRUCTURE_SENSITIVE_RETRY_TASKS
    ) else 1

    for retry_num in range(max_retries):
        if is_valid:
            break
        try:
            retry_prompt = _build_retry_prompt(
                user_prompt, reason, task_name, answer, retry_num
            )
            cot_text, usage2, final_evidence_warnings, final_evidence_stats = _call_with_prompt(retry_prompt)
            result["usage"] = {
                k: result["usage"].get(k, 0) + usage2.get(k, 0)
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
            }
            is_valid, reason = validate_cot(cot_text, answer, task_name)
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"LLM retry {retry_num+1} failed: {e}"
            return result

    if not is_valid:
        result["status"] = "error"
        result["error"] = f"Causal trace validation failed after retry: {reason}"
        return result


    cot_text = cot_text.replace("</think>", "").replace("<think>", "").strip()




    template_count, template_patterns = _count_template_patterns(cot_text)
    if template_count > 3:
        logger.warning(f"[{task_name}] {qid}: Template-like causal trace detected "
                       f"({template_count} patterns: {template_patterns[:5]})")
    cot_text = _strip_heading_artifacts(cot_text)


    enriched = dict(qa_record)
    enriched["conversations"] = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": f"<think>\n{cot_text}\n</think>\n\n{answer}"},
    ]
    enriched_meta = dict(meta)
    enriched_meta["causal_trace"] = {
        "pipeline_id": TRACE_PIPELINE_ID,
        "model": llm.model,
        "reasoning_effort": llm.reasoning_effort,
        "task_family": TASK_FAMILY_BY_NAME.get(task_name, "unknown"),
        "task_contract": get_task_trace_contract(task_name),
        "template_score": template_count,
        "plan_context_available": bool(plan_context),
        "evidence_mode": evidence_mode,
        "max_images": max_images,
        "max_video_frames": max_video_frames,
        "visual_evidence": dict(final_evidence_stats),
        "visual_evidence_attempted_max": dict(attempted_evidence_stats),
        "evidence_warnings": sorted(set(final_evidence_warnings)),
        "evidence_warnings_all_attempts": sorted(set(all_evidence_warnings)),
    }
    enriched["meta"] = enriched_meta

    result["record"] = enriched
    result["status"] = "done"
    return result






def process_task(
    *,
    task_name: str,
    input_dir: str,
    output_dir: str,
    llm: CoTLlm,
    plan_cache: PlanCache,
    parallel_n: int,
    resume: bool,
    keep_going: bool,
    issues: IssueLogger,
    evidence_mode: str,
    max_images: int,
    max_video_frames: int,
    output_filename: str,
) -> Dict[str, int]:

    stats = {"total": 0, "done": 0, "skipped": 0, "errors": 0}


    task_input_dir = os.path.join(input_dir, task_name)
    input_jsonl = os.path.join(task_input_dir, "data.jsonl")
    if not os.path.isfile(input_jsonl):
        logger.warning(f"[{task_name}] No data.jsonl found at {input_jsonl}, skipping.")
        return stats


    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_jsonl = os.path.join(task_output_dir, output_filename)
    resume_path = os.path.join(task_output_dir, "resume_state.jsonl")



    processed_ids: Set[str] = set()
    processed_fingerprints: Dict[str, int] = {}
    if resume:
        processed_fingerprints = load_output_fingerprints(output_jsonl)
        processed_ids = load_resume_state(resume_path)
        processed_rows = sum(processed_fingerprints.values())
        if processed_rows:
            logger.info(f"[{task_name}] Resume: {processed_rows} output rows already written.")
        elif processed_ids:
            logger.info(f"[{task_name}] Resume: {len(processed_ids)} processed ids found.")


    records: List[dict] = []
    parse_errors = 0
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    if parse_errors > 0:
        pct = parse_errors / max(len(records) + parse_errors, 1) * 100
        logger.warning(f"[{task_name}] {parse_errors} malformed JSON lines "
                       f"({pct:.1f}%) in {input_jsonl}")
        if pct > 1.0:
            logger.error(f"[{task_name}] >1% malformed lines — data file may be corrupted")
            issues.log("error", f"High malformed rate: {parse_errors} lines ({pct:.1f}%)",
                       task_name=task_name)

    stats["total"] = len(records)




    if resume:
        remaining_done = defaultdict(int, processed_fingerprints)
        pending = []
        for r in records:
            fp = _qa_instance_fingerprint(r)
            if remaining_done.get(fp, 0) > 0:
                remaining_done[fp] -= 1
            elif not processed_fingerprints and r.get("id", "") in processed_ids:
                continue
            else:
                pending.append(r)
    else:
        pending = list(records)
    stats["skipped"] = len(records) - len(pending)

    if not pending:
        logger.info(f"[{task_name}] All {len(records)} records already processed.")
        return stats

    logger.info(f"[{task_name}] Processing {len(pending)}/{len(records)} records "
                f"({stats['skipped']} resumed), parallel={parallel_n}")


    def _process_one(qa_rec: dict) -> Tuple[dict, Dict[str, Any]]:

        try:
            res = process_one_qa(
                qa_rec,
                llm=llm,
                plan_cache=plan_cache,
                task_name=task_name,
                evidence_mode=evidence_mode,
                max_images=max_images,
                max_video_frames=max_video_frames,
            )
            return qa_rec, res
        except Exception as e:
            return qa_rec, {
                "record": None,
                "status": "error",
                "error": f"Unexpected: {e}\n{traceback.format_exc()}",
                "usage": {},
            }


    _write_lock = threading.Lock()

    def _write_result(qa_rec: dict, res: Dict[str, Any]):

        qid = qa_rec.get("id", str(uuid.uuid4()))
        instance_fp = _qa_instance_fingerprint(qa_rec)
        with _write_lock:
            if res["status"] == "done" and res["record"] is not None:
                append_jsonl(output_jsonl, res["record"])
                append_jsonl(resume_path, {
                    "id": qid,
                    "instance_fingerprint": instance_fp,
                    "status": "done",
                })
                stats["done"] += 1
            elif res["status"] == "error":
                err_msg = res.get("error", "unknown")
                issues.log("error", err_msg, task_name=task_name, qa_id=qid)
                append_jsonl(resume_path, {
                    "id": qid,
                    "instance_fingerprint": instance_fp,
                    "status": "error",
                    "error": err_msg,
                })
                stats["errors"] += 1
                if not keep_going:
                    raise RuntimeError(f"[{task_name}] Error on {qid}: {err_msg}")
            else:
                stats["skipped"] += 1


    log_interval = max(1, len(pending) // 20)                          
    if parallel_n <= 1:
        for i, r in enumerate(pending):
            qa_rec, res = _process_one(r)
            _write_result(qa_rec, res)
            if (i + 1) % log_interval == 0:
                logger.info(f"[{task_name}] Progress: {i+1}/{len(pending)} "
                            f"({stats['done']} ok, {stats['errors']} err)")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_n) as pool:
            future_to_rec = {
                pool.submit(_process_one, r): r for r in pending
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_rec):
                qa_rec, res = future.result()
                _write_result(qa_rec, res)
                completed += 1
                if completed % log_interval == 0:
                    logger.info(f"[{task_name}] Progress: {completed}/{len(pending)} "
                                f"({stats['done']} ok, {stats['errors']} err)")

    logger.info(f"[{task_name}] Done: {stats['done']} ok, {stats['errors']} errors, "
                f"{stats['skipped']} skipped out of {stats['total']} total.")
    logger.info(f"[{task_name}] LLM {llm.token_summary()}")

    return stats






def main():
    parser = argparse.ArgumentParser(
        description="Generate causal reasoning traces for existing QA data.")
    parser.add_argument("--input-dir", required=True,
                        help="Package directory containing Task_XX/data.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for causal-trace-enriched data")
    parser.add_argument("--parallel-api", type=int, default=32,
                        help="Number of concurrent LLM API calls (default: 32)")
    parser.add_argument("--tasks", default="",
                        help="Comma-separated task names or prefixes to process "
                             "(default: all found in input-dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed QA records")
    parser.add_argument("--keep-going", action="store_true", default=True,
                        help="Continue on errors (default: True)")
    parser.add_argument("--no-keep-going", action="store_false", dest="keep_going")
    parser.add_argument("--model", default="gpt-5.4",
                        help="Model deployment name (default: gpt-5.4)")
    parser.add_argument("--reasoning-effort", default="high",
                        help="Reasoning effort level (default: high)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max API retries per call (default: 3)")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="API timeout in seconds (default: 180)")
    parser.add_argument("--evidence-mode", default="auto",
                        choices=["auto", "text_only", "require_readable"],
                        help="Attach readable images/video-sampled frames when possible (default: auto)")
    parser.add_argument("--max-images", type=int, default=4,
                        help="Maximum image evidence items to attach per sample (default: 4)")
    parser.add_argument("--max-video-frames", type=int, default=4,
                        help="Maximum sampled video frames to attach per sample (default: 4)")
    parser.add_argument("--visual-io-parallel", type=int, default=8,
                        help="Maximum concurrent image/video reads and frame extraction jobs "
                             "(default: 8; keeps API concurrency independent from media I/O)")
    parser.add_argument("--output-filename", default="data_causal_trace.jsonl",
                        help="Output JSONL filename inside each Task dir "
                             "(default: data_causal_trace.jsonl; use data.jsonl for a drop-in copy)")
    args = parser.parse_args()

    global _VISUAL_IO_SEMAPHORE
    if args.visual_io_parallel > 0:
        _VISUAL_IO_SEMAPHORE = threading.BoundedSemaphore(args.visual_io_parallel)


    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    available_tasks = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("Task_")
    ])

    if args.tasks:

        requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
        selected_tasks = []
        for req in requested:
            for at in available_tasks:
                if at.startswith(req) or req in at:
                    selected_tasks.append(at)
        selected_tasks = sorted(set(selected_tasks))
    else:
        selected_tasks = available_tasks

    if not selected_tasks:
        logger.error(f"No matching tasks found in {input_dir}. Available: {available_tasks}")
        sys.exit(1)

    logger.info(f"Causal Trace Generation Pipeline")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Tasks:  {len(selected_tasks)} — {', '.join(selected_tasks)}")
    logger.info(f"  Model:  {args.model}  reasoning_effort={args.reasoning_effort}")
    logger.info(f"  Parallel API: {args.parallel_api}")
    logger.info(f"  Resume: {args.resume}")
    logger.info(f"  Keep-going: {args.keep_going}")
    logger.info(f"  Evidence mode: {args.evidence_mode} "
                f"(max_images={args.max_images}, max_video_frames={args.max_video_frames})")
    logger.info(f"  Visual I/O parallel: {args.visual_io_parallel}")
    logger.info(f"  Output filename: {args.output_filename}")


    llm = CoTLlm(
        model=args.model,
        max_retries=args.max_retries,
        retry_backoff=5.0,
        timeout_sec=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )
    plan_cache = PlanCache()
    issues = IssueLogger(output_dir)


    all_stats = defaultdict(int)
    t0 = time.time()

    for task_name in selected_tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {task_name}")
        logger.info(f"{'='*60}")


        plan_cache._cache.clear()

        try:
            stats = process_task(
                task_name=task_name,
                input_dir=input_dir,
                output_dir=output_dir,
                llm=llm,
                plan_cache=plan_cache,
                parallel_n=args.parallel_api,
                resume=args.resume,
                keep_going=args.keep_going,
                issues=issues,
                evidence_mode=args.evidence_mode,
                max_images=args.max_images,
                max_video_frames=args.max_video_frames,
                output_filename=args.output_filename,
            )
            for k, v in stats.items():
                all_stats[k] += v
        except Exception as e:
            logger.error(f"[{task_name}] Fatal error: {e}")
            if not args.keep_going:
                raise
            issues.log("error", f"Task-level failure: {e}", task_name=task_name)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Causal Trace Generation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Total records: {all_stats['total']}")
    logger.info(f"  Done:          {all_stats['done']}")
    logger.info(f"  Errors:        {all_stats['errors']}")
    logger.info(f"  Skipped:       {all_stats['skipped']}")
    logger.info(f"  Elapsed:       {elapsed/60:.1f} min")
    logger.info(f"  LLM summary:   {llm.token_summary()}")
    logger.info(f"  Issues:        {issues.error_count} errors, {issues.warning_count} warnings")


if __name__ == "__main__":
    main()
