




from __future__ import annotations

import argparse
import atexit
import urllib.error
import urllib.request
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import time
import traceback
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_stage_one_qa")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


TASK_01 = "Task_01_Spatial_Precondition"
TASK_02 = "Task_02_Affordance_Precondition"
TASK_03 = "Task_03_Physical_Feasibility"
TASK_04 = "Task_04_Affordance_Visual_Semantics"
TASK_05 = "Task_05_Holistic_Causal_Chain"
TASK_06 = "Task_06_Spatial_Postcondition"
TASK_07 = "Task_07_Affordance_Postcondition"
TASK_08 = "Task_08_Goal_Recognition"
TASK_09 = "Task_09_Macro_Anchor_Extraction"
TASK_10 = "Task_10_Clip_to_Step_Goal"
TASK_11 = "Task_11_Action_Phrase"
TASK_12 = "Task_12_State_Evolution"
TASK_13 = "Task_13_Strategic_Rationale"
TASK_14 = "Task_14_Inter_Step_Dependency"
TASK_15 = "Task_15_Next_Step_Prediction"
TASK_16 = "Task_16_Middle_Steps_Infill"
TASK_17 = "Task_17_Next_K_Steps_Prediction"
TASK_18 = "Task_18_Bad_Plan_Diagnosis_And_Repair"
TASK_19 = "Task_19_Counterfactual_Outcome"
TASK_20 = "Task_20_Failure_Recovery"
INTERNAL_PATIENT_IDENTIFICATION_TASK = "Internal_Patient_Identification"
INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK = "Internal_Hotspot_Affordance_Type"
INTERNAL_HOTSPOT_MECHANISM_TASK = "Internal_Hotspot_Mechanism"
INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK = "Internal_Next_Step_After_Recovery"


ALL_TASKS: Tuple[str, ...] = (
    TASK_01,
    TASK_02,
    TASK_03,
    TASK_04,
    TASK_05,
    TASK_06,
    TASK_07,
    TASK_08,
    TASK_09,
    TASK_10,
    TASK_11,
    TASK_12,
    TASK_13,
    TASK_19,
    TASK_20,
)


DEFAULT_TASKS: Tuple[str, ...] = (
    TASK_01,
    TASK_02,
    TASK_03,
    TASK_04,
    TASK_05,
    TASK_06,
    TASK_07,
    TASK_08,
    TASK_09,
    TASK_10,
    TASK_11,
    TASK_12,
    TASK_13,
    TASK_19,
    TASK_20,
)


EVIDENCE_KEYFRAME = "keyframe_single"
EVIDENCE_UNIFORM = "images_uniform_scene"
EVIDENCE_CLIP = "video_clip"
EVIDENCE_PREFIX = "video_prefix"

_FRAME_TS_CACHE: Dict[Tuple[str, int], Dict[int, float]] = {}


FRAME_LEAK_PATTERNS = [
    re.compile(r"\bframe_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bsample_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bts_\d", re.IGNORECASE),
    re.compile(r"\.(jpg|jpeg|png|mp4)\b", re.IGNORECASE),
    re.compile(r"\b(frame|image)\s*\d+\b", re.IGNORECASE),
]


_KEY_MOMENT_PREFIX_RE = re.compile(r"^\s*Key moment\s*\d+\s*\([^)]*\)\s*:\s*", re.IGNORECASE)
_TS_RE = re.compile(r"_ts_(\d+(?:\.\d+)?)s\b", re.IGNORECASE)
_SNAKE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_LOWER_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]*\b")
_STANDALONE_NOT_OBSERVABLE_RE = re.compile(r"^\(?\s*not\s+directly\s+observable\s*\)?\.?$", re.IGNORECASE)


_GENERIC_OBJECT_TOKENS = {
    "hand",
    "hands",
    "left_hand",
    "right_hand",
    "gloved_left_hand",
    "gloved_right_hand",
    "bare_left_hand",
    "bare_right_hand",
    "robotic_hand",
    "robotic_hands",
    "person",
    "human",
    "agent",
    "body",
    "arm",
    "finger",
    "fingers",
}


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "is",
    "are",
    "be",
    "by",
    "for",
    "with",
    "then",
    "into",
    "from",
    "that",
    "this",
    "it",
    "as",
    "after",
    "before",
    "when",
    "while",
    "so",
    "can",
    "cannot",
    "not",
    "no",
    "yes",
    "will",
    "would",
    "should",
    "could",
    "must",
    "may",
    "might",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "have",
    "has",
    "had",
}


_VERBISH_TOKENS = {
    "aligned",
    "contact",
    "contacting",
    "supports",
    "support",
    "supported",
    "stabilize",
    "stabilized",
    "stabilizing",
    "position",
    "positioned",
    "place",
    "placed",
    "placing",
    "remove",
    "removed",
    "removing",
    "open",
    "opened",
    "opening",
    "close",
    "closed",
    "closing",
    "tilt",
    "tilted",
    "tilting",
    "pour",
    "pours",
    "pouring",
    "lift",
    "lifted",
    "lifting",
    "rotate",
    "rotates",
    "rotating",
    "hold",
    "holds",
    "holding",
    "begin",
    "begins",
    "beginning",
    "start",
    "starts",
    "starting",
}

_DISTRACTOR_OBJECTS = [
    "microwave",
    "dish_soap",
    "flashlight",
    "hammer",
    "screwdriver",
    "toothbrush",
    "laptop",
    "phone",
    "remote_control",
    "shampoo_bottle",
    "shoe",
    "book",
]


_RELATION_TOKENS = {
    "on_top_of",
    "inside",
    "inside_of",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "above",
    "below",
    "next_to",
    "relative_to",
    "separated_from",
    "connected_to",
    "disconnected_from",
    "aligned_with",
    "tilted_toward",
    "centered_on",
    "filled_with",
    "covered_by",
    "supported_by",
    "stabilized_on",
    "contacting",
}


_GENERIC_PART_TOKENS = {
    "opening",
    "mouth",
    "rim",
    "edge",
    "surface",
    "point",
    "region",
    "area",
    "contact",
    "grip",
    "handle",
}


@dataclass(frozen=True)
class Sample:
    task_name: str
    evidence_type: str
    image: List[str]
    video: Optional[str]
    question: str
    answer: str
    source_path: str
    llm_fields: Optional[Dict[str, Any]] = None


@dataclass
class ApiConfig:
    api_key: str = os.environ.get("API_KEY", "EMPTY")
    api_base_url: str = os.environ.get("API_BASE_URL", "")
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "openai_compatible")
    model_name: str = os.environ.get("MODEL_NAME", "gpt-5.4")
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "4096"))
    request_images_limit: int = int(os.environ.get("REQUEST_IMAGES_LIMIT", "1000000"))
    max_retries: int = int(os.environ.get("MAX_RETRIES", "3"))
    retry_backoff_sec: float = float(os.environ.get("RETRY_BACKOFF_SEC", "1.5"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.3"))


def initialize_api_client(cfg: ApiConfig) -> Any:


    return cfg


SYSTEM_PROMPT = """You are an expert Physical Interaction Analyst and Physics Consultant.
Your task is to rewrite draft QA answers into fluent, professional English using ONLY the provided SOURCE_JSON fields.

Core objectives:
1) Naturalness: do not just list fields; connect them into a coherent answer.
2) Strict grounding: do NOT add any new objects, actions, states, or causal claims beyond SOURCE_JSON.
3) Detail & rigor: preserve all technical details; do not simplify away constraints/statuses.
4) Professional tone: objective, academic; no conversational filler.
5) Task compliance: obey the task-specific formatting/constraints exactly (e.g., one sentence, no newlines, required phrases).
6) Minimal-risk editing: if the draft already satisfies all constraints and is clear, output it unchanged.
7) Lexical fidelity: prefer reusing key nouns/verbs from SOURCE_JSON/DRAFT_TEXT; avoid introducing new concrete noun phrases.

Domain definitions & grounding rules (use consistently):
- Spatial relations = directly observable geometry/topology/contact/state (e.g., in contact with, on top of, inside, aligned with, supported by, open/closed).
  * A good spatial statement explicitly names two entities and their visible relation.
  * Avoid abstract/non-visual claims like "accessible/within reach/convenient" unless grounded by a visible metric (e.g., distance, alignment).
  * Do NOT invent numeric measurements (distances/angles/weights) unless explicitly present in SOURCE_JSON or DRAFT_TEXT.
- Affordances = operability/readiness states directly visible or strongly implied by mechanical state (e.g., open/closed, sealed/unsealed, empty/full, blocked/unblocked, graspable/not graspable, stable/unstable, separated/clumped).
  * Do NOT assert hidden qualities ("sharp", "clean", "functional heater", "active heat") unless clearly visible in the evidence encoded in SOURCE_JSON.
  * Material properties (e.g., sharp/hot/clean) are allowed ONLY when explicitly stated in SOURCE_JSON (treated as visually confirmed upstream).
  * Do NOT replace affordances with high-level semantic goals ("ready to be cooked/served").
  * Do NOT introduce "ready to ..." affordance phrases unless they are explicitly supported by SOURCE_JSON or already present in the draft.
- Mechanism = concise physical explanation grounded in contacts/forces (force/torque transfer, friction, leverage, constraint satisfaction, flow).

FORMAT STANDARD for multi-sentence causal statements (when applicable):
- Each statement must be a single, complete, objective English sentence.
- Each statement must end with '.'.
- Do NOT start any sentence with a list marker or numbering (unless the draft already contains required numbering for the task).
- Do NOT use newline characters inside the final answer (unless the task explicitly requires multiple paragraphs).

Conflict resolution:
- Treat SOURCE_JSON as the ground truth. If DRAFT_TEXT conflicts with SOURCE_JSON, correct it using SOURCE_JSON while staying strictly within SOURCE_JSON.
- If the draft contains "not directly observable", do NOT turn that claim into a definite statement.

Dataset safety:
- Do NOT mention filenames, paths, extensions, timestamps, or frame numbers.
- Do NOT mention JSON keys/field names or section headers; output answer content only.
- Do NOT output markdown or code fences.
- Avoid placeholders like "unknown", "N/A", or "...".
- Do NOT use first-person claims ("I think", "we can see"); state facts directly or use "not directly observable" when needed.

Output format:
- Return ONLY the final answer text.
"""


TWO_PARAGRAPH_TASKS: set[str] = set()
SINGLE_SENTENCE_TASKS = {TASK_05, TASK_03}


LLM_FLEX_TASKS: set[str] = {

    TASK_04,
    TASK_05,
    TASK_14,
    TASK_20,
}




LLM_RELAXED_SPAN_TASKS: set[str] = set(LLM_FLEX_TASKS) | {
    INTERNAL_HOTSPOT_MECHANISM_TASK,
    TASK_12,
    TASK_13,
    TASK_01,
    TASK_02,
    TASK_03,
    TASK_06,
    TASK_07,
    TASK_19,
}



STRICT_DRAFT_QUOTE_TASKS: set[str] = {TASK_18}





LLM_SKIP_TASKS: set[str] = set()


DEFAULT_LLM_TASKS: Tuple[str, ...] = ALL_TASKS


def _sanitize_text_single_line(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_answer(task_name: str, text: str) -> str:
    if not text:
        return ""
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"```[a-zA-Z]*\s*", "", s)
    s = s.replace("```", "")
    s = re.sub(r"(?m)^\s*([\-\*•\>]+|\d+[\.\)])\s+", "", s)
    s = s.strip()
    if task_name in TWO_PARAGRAPH_TASKS:
        parts = re.split(r"\n\s*\n", s)
        parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts[0] + "\n\n" + "\n\n".join(parts[1:2])
        return _sanitize_space(s)
    return _sanitize_text_single_line(s)


def _defluff_text(text: str) -> str:
    s = str(text or "")
    patterns = [
        r"^\s*(In summary|In conclusion|To summarize|Overall|In general|Generally),\s*",
        r"^\s*(In this (scene|image|frame|step)),\s*",
        r"^\s*(It should be noted that|Note that)\s*",
        r"^\s*(Here is the answer|Answer)\s*[:\-]\s*",
    ]
    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_DOUBLE_QUOTED_RE = re.compile(r"\"([^\"]+)\"")


def _extract_double_quoted(text: str) -> List[str]:
    s = str(text or "")
    return [m.group(1) for m in _DOUBLE_QUOTED_RE.finditer(s)]


def _answer_mentions_token(answer: str, token: str) -> bool:

    t = str(token or "").strip()
    if not t:
        return False
    a = str(answer or "")
    for q in _extract_double_quoted(a):
        if str(q).strip() == t:
            return True

    try:
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", a) is not None
    except re.error:                    
        return t in a


def _repair_double_quoted_from_draft(*, draft: str, candidate: str) -> Tuple[str, bool]:


    draft_quotes = [q for q in _extract_double_quoted(draft) if q]
    if not draft_quotes:
        return candidate, False

    cand = str(candidate or "")
    matches = list(_DOUBLE_QUOTED_RE.finditer(cand))
    if len(matches) != len(draft_quotes):
        return candidate, False

    cand_quotes = [m.group(1) for m in matches]
    if cand_quotes == draft_quotes:
        return candidate, False

    parts: List[str] = []
    last = 0
    for i, m in enumerate(matches):
        parts.append(cand[last : m.start(1)])
        parts.append(draft_quotes[i])
        last = m.end(1)
    parts.append(cand[last:])
    return "".join(parts), True


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
        return
    if isinstance(obj, (list, tuple)):
        for it in obj:
            yield from _iter_strings(it)
        return


def _collect_snake_tokens(fields: Any, *, extra_texts: Optional[Sequence[str]] = None) -> set[str]:
    toks: set[str] = set()
    for s in _iter_strings(fields):
        toks.update(_SNAKE_TOKEN_RE.findall(str(s)))
    for t in extra_texts or []:
        toks.update(_SNAKE_TOKEN_RE.findall(str(t or "")))
    return toks


def _extract_task10_pre_eff(draft_answer: str) -> Tuple[str, str]:
    s = _sanitize_text_single_line(draft_answer)
    pre = ""
    eff = ""
    m = re.match(r"^When\s+(.+?),\s+", s)
    if m:
        pre = m.group(1).strip()
    idx = s.lower().rfind("because ")
    if idx != -1:
        after = s[idx + len("because ") :].strip()
        parts = after.split(",", 1)
        if len(parts) == 2:
            eff = parts[1].strip().rstrip(".?!").strip()
    return pre, eff


def _extract_task17_eff_pre(draft_answer: str) -> Tuple[str, str]:
    s = _sanitize_text_single_line(draft_answer)
    eff = ""
    pre = ""
    m = re.search(r"\bestablishes\s+(.*?),\s+thereby\b", s, flags=re.IGNORECASE)
    if m:
        eff = m.group(1).strip()
    m2 = re.search(r"\bensure\s+(.*?)[.?!]?\s*$", s, flags=re.IGNORECASE)
    if m2:
        pre = m2.group(1).strip()
    return eff, pre


def _llm_required_spans(task_name: str, *, fields: Dict[str, Any], draft_answer: str) -> List[str]:
    spans: List[str] = []

    def _add(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                spans.append(s)
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _add(x)

    if task_name == TASK_08:
        _add(fields.get("high_level_goal"))
    elif task_name == TASK_09:
        _add(fields.get("key_objects"))
    elif task_name == TASK_10:
        _add(fields.get("step_goal"))
    elif task_name == INTERNAL_PATIENT_IDENTIFICATION_TASK:
        _add(fields.get("patient"))
    elif task_name == TASK_11:
        _add(fields.get("action"))
    elif task_name == INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK:
        _add(fields.get("affordance_type"))
    elif task_name == INTERNAL_HOTSPOT_MECHANISM_TASK:
        _add(fields.get("mechanism"))
    elif task_name == TASK_04:
        desc = str(fields.get("hotspot_description") or "").strip()
        mech = str(fields.get("mechanism") or "").strip()
        if desc:
            spans.append(_lowercase_first_alpha(desc.strip().rstrip(".")))
        if mech:
            spans.append(_lowercase_first_alpha(_inline_clause(mech)))
    elif task_name == TASK_12:
        _add(fields.get("action_state_change_description"))
    elif task_name == TASK_05:
        pre, eff = _extract_task10_pre_eff(draft_answer)
        if pre:
            spans.append(pre)
        if eff:
            spans.append(eff)
    elif task_name == TASK_13:
        _add(fields.get("rationale"))
    elif task_name == TASK_01:
        _add(fields.get("spatial_preconditions"))
    elif task_name == TASK_02:
        _add(fields.get("affordance_preconditions"))
    elif task_name == TASK_03:
        _add(fields.get("spatial_precondition"))
        _add(fields.get("affordance_precondition"))
    elif task_name == TASK_06:
        _add(fields.get("spatial_postconditions"))
    elif task_name == TASK_07:
        _add(fields.get("affordance_postconditions"))
    elif task_name == TASK_14:
        eff, pre = _extract_task17_eff_pre(draft_answer)
        if eff:
            spans.append(eff)
        if pre:
            spans.append(pre)
    elif task_name == TASK_15:
        _add(fields.get("next_step_goal"))
    elif task_name == TASK_16:
        _add(fields.get("middle_step_goals"))
    elif task_name == TASK_17:
        _add(fields.get("next_step_goals"))
    elif task_name == TASK_18:
        flaw_step = fields.get("flaw_step")
        flaw_type = fields.get("flaw_type")
        try:
            flaw_step_i = int(flaw_step) if flaw_step is not None else 0
        except Exception:
            flaw_step_i = 0
        flaw_type_s = str(flaw_type or "").strip()
        if flaw_step_i > 0:
            spans.append(f"FlawStep={flaw_step_i}")
        if flaw_type_s:
            spans.append(f"FlawType={flaw_type_s}")
        _add(fields.get("repair_steps"))
    elif task_name == TASK_19:
        _add(fields.get("expected_outcome"))
    elif task_name == TASK_20:
        strat = str(fields.get("recovery_strategy") or "").strip()
        if strat:
            spans.append(strat.rstrip(".!?").strip())
    elif task_name == INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK:
        _add(fields.get("next_step_goal"))


    if not spans:
        d = str(draft_answer or "").strip()
        if d:
            spans.append(d)

    return [s for s in spans if s]


def _term_overlap_count(required: set[str], cand: set[str]) -> int:
    if not required or not cand:
        return 0
    cand_list = sorted(cand)
    matched = 0
    for t in required:
        if t in cand:
            matched += 1
            continue
        if len(t) < 4:
            continue
        for c in cand_list:
            if len(c) < 4:
                continue
            pref = os.path.commonprefix([t, c])
            if pref and len(pref) >= 4:
                matched += 1
                break
    return matched


def _span_present_in_text(span: str, text: str, *, relaxed: bool) -> bool:

    s = _sanitize_text_single_line(span).lower()
    if not s:
        return True
    t = _sanitize_text_single_line(text).lower()
    if s in t:
        return True
    if not bool(relaxed):
        return False
    span_terms = _normalize_terms(s)
    if not span_terms:
        return False
    text_terms = _normalize_terms(t)
    if not text_terms:
        return False
    matched = _term_overlap_count(span_terms, text_terms)
    n = len(span_terms)


    need = max(1, n - 1) if n <= 3 else max(2, int(math.ceil(0.30 * n)))
    need = min(8, need)
    return matched >= need


def _llm_output_is_acceptable(
    *,
    task_name: str,
    draft_answer: str,
    fields: Dict[str, Any],
    candidate: str,
    question: Optional[str] = None,
    vocab_guard: str = "off",
    verify: str = "warn",
) -> Tuple[bool, str]:
    out = str(candidate or "").strip()
    if not out:
        return False, "empty"
    if _has_frame_leak(out):
        return False, "frame/file leak"

    lower = out.lower()
    if "###" in out or "input data" in lower or "ground truth" in lower or "draft answer" in lower or "polished answer" in lower:
        return False, "prompt-echo"

    verify_s = str(verify or "strict").strip().lower()
    if verify_s not in {"strict", "warn", "off"}:
        verify_s = "strict"

    vocab_guard_s = str(vocab_guard or "strict").strip().lower()
    if vocab_guard_s not in {"strict", "warn", "off"}:
        vocab_guard_s = "strict"

    warn_reasons: List[str] = []

    def _fail_or_warn(reason: str) -> Optional[Tuple[bool, str]]:
        if verify_s == "strict":
            return False, reason
        if verify_s == "warn":
            warn_reasons.append(str(reason))
        return None


    if task_name == TASK_18:
        s = _sanitize_text_single_line(out)
        if not re.match(
            r"^FlawStep=\d+;\s*FlawType=[a-z_]+;\s*Reason=.+?;\s*Repair:\s*1\)\s*\".+\"\s+2\)\s*\".+\"\s+3\)\s*\".+\"\s*$",
            s,
        ):
            return False, "bad Task_18 format"

    def _extract_task02_candidates(q: str) -> List[str]:

        text = str(q or "")
        if not text.strip():
            return []
        lower = text.lower()
        i = lower.find("candidate objects")
        if i < 0:
            return []
        b = text.find("[", i)
        if b < 0:
            return []
        depth = 0
        in_str = False
        esc = False
        for j in range(b, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == "[":
                depth += 1
                continue
            if ch == "]":
                depth -= 1
                if depth == 0:
                    raw = text[b : j + 1]
                    try:
                        arr = json.loads(raw)
                    except Exception:
                        return []
                    if not isinstance(arr, list):
                        return []
                    out: List[str] = []
                    for x in arr:
                        if isinstance(x, str) and x.strip():
                            out.append(x.strip())
                    return out
        return []

    def _mentions_token(text_lower: str, token: str) -> bool:
        tl = str(token or "").strip().lower()
        if not tl:
            return False

        if re.search(rf"\b{re.escape(tl)}\b", text_lower):
            return True

        if "_" in tl:
            parts = [re.escape(p) for p in tl.split("_") if p]
            if parts:
                pat = r"\b" + r"\s+".join(parts) + r"\b"
                if re.search(pat, text_lower):
                    return True
        return False

    if verify_s != "off":

        relaxed_span = task_name in LLM_RELAXED_SPAN_TASKS
        for span in _llm_required_spans(task_name, fields=fields, draft_answer=draft_answer):
            if not _span_present_in_text(span, out, relaxed=bool(relaxed_span)):
                res = _fail_or_warn(f"missing required span: {str(span)[:60]!r}")
                if res is not None:
                    return res


        required_snake = _collect_snake_tokens({}, extra_texts=[draft_answer])
        allowed_snake = _collect_snake_tokens(fields, extra_texts=[draft_answer])
        cand_snake = set(_SNAKE_TOKEN_RE.findall(out))
        missing = sorted([t for t in required_snake if t not in cand_snake])
        if missing:
            res = _fail_or_warn(f"missing required snake_case tokens: {missing[:8]}")
            if res is not None:
                return res
        extra = sorted([t for t in cand_snake if t not in allowed_snake])
        if extra:
            res = _fail_or_warn(f"introduced unknown snake_case tokens: {extra[:8]}")
            if res is not None:
                return res


        draft_quotes = [q for q in _extract_double_quoted(draft_answer) if q]
        if task_name in STRICT_DRAFT_QUOTE_TASKS:
            for q in draft_quotes:

                if q not in out:
                    res = _fail_or_warn(f"missing quoted span: {q[:60]!r}")
                    if res is not None:
                        return res

        cand_quotes = [q for q in _extract_double_quoted(out) if q]
        if task_name in STRICT_DRAFT_QUOTE_TASKS:
            extra_quotes = sorted({q for q in cand_quotes if q not in set(draft_quotes)})
            if extra_quotes:
                res = _fail_or_warn(f"introduced new quoted span: {extra_quotes[0][:60]!r}")
                if res is not None:
                    return res


    if task_name == TASK_09:
        ko_raw = fields.get("key_objects")
        key_objects = set()
        if isinstance(ko_raw, (list, tuple)):
            for x in ko_raw:
                if isinstance(x, str) and x.strip():
                    key_objects.add(x.strip().lower())
        lower_out = out.lower()
        candidates = _extract_task02_candidates(question or "")
        if candidates:
            for c in candidates:
                cl = str(c or "").strip().lower()
                if not cl:
                    continue
                if cl in key_objects:
                    continue
                if _mentions_token(lower_out, cl):
                    return False, f"mentioned non-key candidate object: {c}"

        for d in _DISTRACTOR_OBJECTS:
            dl = str(d or "").strip().lower()
            if not dl or "_" in dl:
                continue
            if dl in key_objects:
                continue
            if re.search(rf"\b{re.escape(dl)}\b", lower_out):
                return False, f"mentioned distractor object: {d}"





    novel_warn_terms: List[str] = []
    if vocab_guard_s != "off":
        extra_ok_terms = {

            "because",
            "therefore",
            "thus",
            "hence",
            "thereby",
            "overall",
            "specifically",
            "however",
            "although",
            "though",
            "instead",
            "rather",
            "otherwise",
            "moreover",
            "furthermore",
            "additionally",
            "alternatively",
            "meanwhile",
            "whereas",
            "since",

            "will",
            "would",
            "should",
            "could",
            "must",
            "might",
            "cannot",
            "does",
            "done",
            "doing",

            "include",
            "includes",
            "including",
            "list",
            "lists",
            "listed",
            "listing",
            "item",
            "items",
            "object",
            "objects",
            "relevant",
            "necessary",
            "needed",
            "required",
            "primary",
            "main",
            "essential",
            "integral",
            "various",
            "several",
            "namely",
            "generally",
            "general",
            "ultimately",

            "also",
            "well",
            "such",
            "other",
            "another",
            "both",
            "which",
            "where",
            "who",
            "whom",
            "whose",
            "they",
            "them",
            "their",
            "theirs",
            "its",
            "there",
            "being",
            "been",
            "have",
            "has",
            "had",
            "use",
            "uses",
            "used",
            "using",
            "achieve",
            "achieves",
            "achieved",
            "achieving",
            "involve",
            "involves",
            "involved",
            "involving",
            "represent",
            "represents",
            "represented",
            "representing",
            "identify",
            "identifies",
            "identified",
            "identifying",
            "focus",
            "focuses",
            "focused",
            "focusing",
            "via",
            "within",
            "between",
            "among",
            "around",
            "across",
            "through",
            "throughout",
            "during",
            "upon",
            "along",
            "high-level",
            "initial",
            "final",
            "subsequent",
            "successful",
            "intended",
            "given",
            "possible",
            "defined",
            "definition",
            "whereby",
            "onto",

            "goal",
            "goals",
            "step",
            "steps",
            "state",
            "states",
            "hotspot",
            "interaction",
            "affordance",
            "affordances",
            "mechanism",
            "mechanisms",
            "precondition",
            "preconditions",
            "postcondition",
            "postconditions",
            "effect",
            "effects",
            "outcome",
            "recovery",
            "strategy",
            "strategies",

            "physical",
            "function",
            "functions",
            "provide",
            "provides",
            "providing",
            "utilize",
            "utilizes",
            "utilizing",
            "serve",
            "serves",
            "serving",
            "underlying",
            "consequently",
            "leverage",
            "leverages",
            "leveraging",
            "satisfy",
            "satisfies",
            "satisfying",
            "proceed",
            "proceeds",
            "proceeding",
            "process",
            "processing",

            "force",
            "forces",
            "friction",
            "pressure",
            "motion",
            "contact",
            "signal",
            "activate",
            "activation",
            "trigger",
            "triggers",
            "triggering",
            "enable",
            "enables",
            "enabling",
            "allow",
            "allows",
            "causes",
            "cause",
            "leading",
            "leads",
            "result",
            "results",
            "resulting",

            "feasible",
            "likely",
            "observable",
            "directly",
            "not",
        }
        allowed_terms = _normalize_terms(str(draft_answer or "") + " " + json.dumps(fields or {}, ensure_ascii=False))
        cand_terms = _normalize_terms(out)

        allowed_list = sorted(allowed_terms)

        def _term_allowed(t: str) -> bool:
            if not t:
                return True
            if t in allowed_terms or t in extra_ok_terms:
                return True

            if len(t) < 4:
                return True
            for a in allowed_list:
                if len(a) < 4:
                    continue

                pref = os.path.commonprefix([t, a])
                if pref and len(pref) >= 4:
                    return True
            return False

        novel = sorted([t for t in cand_terms if not _term_allowed(t)])
        if novel:
            if vocab_guard_s == "warn":
                novel_warn_terms = novel[:8]
            else:
                return False, f"introduced novel terms: {novel[:8]}"

    if verify_s != "off":


        if len(out) > max(180, int(4.0 * len(str(draft_answer or ""))) + 220):
            res = _fail_or_warn("too long")
            if res is not None:
                return res


        if task_name in SINGLE_SENTENCE_TASKS:
            if "\n" in out:
                res = _fail_or_warn("must be single line")
                if res is not None:
                    return res

            if len(re.findall(r"[.?!]", out)) > 1:
                res = _fail_or_warn("must be one sentence")
                if res is not None:
                    return res

    if novel_warn_terms:
        warn_reasons.append(f"novel terms: {novel_warn_terms}")
    if warn_reasons and verify_s == "warn":
        uniq: List[str] = []
        for w in warn_reasons:
            if w not in uniq:
                uniq.append(w)
        msg = "; ".join(uniq[:2])
        if len(uniq) > 2:
            msg += f" (+{len(uniq) - 2} more)"
        return True, f"warn: {msg}"
    return True, "ok"


class TwoStageLlm:
    def __init__(self, cfg: ApiConfig):
        self.cfg = cfg
        self.client = initialize_api_client(cfg)

    def enabled(self) -> bool:
        return bool(self.client) and bool(self.cfg.api_base_url) and (self.cfg.api_key or "EMPTY") != "EMPTY"

    def _endpoint(self) -> str:
        base = str(self.cfg.api_base_url or "").strip().rstrip("/")
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = str(self.cfg.api_key or "").strip()
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        prov = str(self.cfg.model_provider_id or "").strip()
        if prov:
            headers["X-Model-Provider-Id"] = prov
        return headers

    def call(
        self,
        *,
        system_prompt: str,
        user_text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        require_success: bool = False,
    ) -> str:
        if not self.enabled():
            if bool(require_success):
                raise RuntimeError("LLM is not enabled (missing API key or API base URL).")
            return ""
        messages = [
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": str(user_text or "")},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(self.cfg.max_retries))):
            try:
                payload = {
                    "model": str(self.cfg.model_name or ""),
                    "messages": messages,
                    "max_tokens": int(max_tokens or self.cfg.max_tokens),
                    "temperature": float(self.cfg.temperature if temperature is None else temperature),
                    "top_p": 0.9,
                    "presence_penalty": 0,
                    "stream": False,
                }
                endpoint = self._endpoint()
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers=self._headers(),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                resp = json.loads(data.decode("utf-8", errors="replace") or "{}")
                choices = resp.get("choices") if isinstance(resp, dict) else None
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError(f"Empty response or missing choices (keys={sorted(resp.keys()) if isinstance(resp, dict) else type(resp)})")
                choice0 = choices[0] if isinstance(choices[0], dict) else {}
                msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
                content = msg.get("content")
                out = ""
                if isinstance(content, str):
                    out = content
                elif isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str) and t.strip():
                                parts.append(t)
                    out = "\n".join(parts).strip()
                out = str(out or "")
                if bool(require_success) and not out.strip():
                    raise RuntimeError("LLM returned empty content.")
                logger.info(f">>> [LLM] ok len={len(out)} endpoint={endpoint}")
                return out
            except urllib.error.HTTPError as e:                    
                last_err = e
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                logger.warning(f">>> [LLM] HTTPError attempt={attempt+1} code={getattr(e, 'code', '?')} body={body[:400]}")
            except Exception as e:                    
                last_err = e
                logger.warning(f">>> [LLM] failed attempt={attempt+1}: {e}")
                if attempt + 1 < max(1, int(self.cfg.max_retries)):
                    time.sleep(float(self.cfg.retry_backoff_sec) * (attempt + 1))
        if last_err is not None:
            logger.error(f">>> [LLM] final failure: {last_err}")
            if bool(require_success):
                raise RuntimeError(f"LLM call failed after {max(1, int(self.cfg.max_retries))} attempts: {last_err}")
        return ""

    def generate_answer(
        self,
        *,
        task_name: str,
        fields: Dict[str, Any],
        draft_answer: str,
        two_pass: bool,
        policy: str = "polish",
        require_success: bool = False,
    ) -> str:
        pol0 = str(policy or "polish").strip().lower()
        if pol0 not in {"polish", "copyedit", "verbatim"}:
            pol0 = "polish"


        if pol0 == "verbatim":
            return str(draft_answer or "")

        required_snake = sorted(_collect_snake_tokens({}, extra_texts=[draft_answer]))
        allowed_snake = sorted(_collect_snake_tokens(fields, extra_texts=[draft_answer]))
        required_quotes = sorted({q for q in _extract_double_quoted(draft_answer) if q})
        required_spans = _llm_required_spans(task_name, fields=fields, draft_answer=draft_answer)
        must_keep_phrase = "not directly observable" if "not directly observable" in str(draft_answer or "").lower() else ""
        strict_quotes = task_name in STRICT_DRAFT_QUOTE_TASKS

        fmt = (
            "Produce exactly TWO paragraphs."
            if task_name in TWO_PARAGRAPH_TASKS
            else (
                "Produce exactly ONE sentence (no newlines)."
                if task_name in SINGLE_SENTENCE_TASKS
                else "Produce a single paragraph."
            )
        )

        def _prompt(draft: str, *, pol: str) -> str:
            action = "Lightly copy-edit" if pol == "copyedit" else "Rewrite"
            span_rule = (
                "Cover REQUIRED_FACTS faithfully. Prefer to keep REQUIRED_FACTS verbatim as substrings (especially label-like fields such as "
                "goals, step-goal labels, identifiers, and fixed-status phrases); you may paraphrase surrounding glue text, but do not omit or alter "
                "the required facts."
            )
            quote_rule = (
                "Preserve any text inside double quotes EXACTLY, and do NOT introduce any new double-quoted strings."
                if bool(strict_quotes)
                else "If the draft contains double-quoted strings, keep their CONTENT unchanged (quote marks optional); avoid introducing new double-quoted strings."
            )
            task_guidelines_lines: List[str] = []
            if task_name == TASK_08:
                task_guidelines_lines.append(
                    "Goal label: output ONLY the high-level goal text as-is (no paraphrase, no extra words, no quotes)."
                )
            if task_name == TASK_09:
                task_guidelines_lines.append(
                    "Object selection: mention ONLY the provided key objects; do NOT mention any other candidate objects/distractors; do NOT add explanations."
                )
            if task_name == TASK_10:
                task_guidelines_lines.append(
                    "Step-goal label: output ONLY the step goal text as-is (no paraphrase, no extra words, no quotes)."
                )
            if task_name == INTERNAL_PATIENT_IDENTIFICATION_TASK:
                task_guidelines_lines.append(
                    "Patient identification: output exactly one short sentence using the provided patient identifier verbatim; do not add any other objects."
                )
            if task_name == TASK_11:
                task_guidelines_lines.append(
                    "Action phrase: output exactly one short sentence using the provided action phrase verbatim; do not paraphrase it."
                )
            if task_name in (INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK,):
                task_guidelines_lines.append(
                    "Affordance type: output exactly one short sentence using the provided affordance-type identifier token verbatim; do NOT paraphrase it into adjectives."
                )
            if task_name in (INTERNAL_HOTSPOT_MECHANISM_TASK, TASK_04):
                task_guidelines_lines.append(
                    "Hotspot mechanism: explain a physical mechanism grounded in contacts/forces (force/torque transfer, friction, leverage, flow, constraint satisfaction); avoid high-level intent/semantic goals."
                )
            if task_name == INTERNAL_HOTSPOT_MECHANISM_TASK:
                task_guidelines_lines.append(
                    "Mechanism output: keep it concise (1-2 sentences) and consistent with the provided mechanism text; do NOT introduce new objects or steps."
                )
            if task_name == TASK_04:
                task_guidelines_lines.append(
                    "Hotspot summary: preserve the hotspot description phrase, affordance-type identifier, and mechanism meaning from the draft; prefer the two-sentence structure: (1) locate hotspot, (2) affordance type + mechanism."
                )
            if task_name == TASK_12:
                task_guidelines_lines.append(
                    "State change description: keep it objective and grounded; do NOT introduce time/frame references or speculative hidden states."
                )
            if task_name in (TASK_01, TASK_06):
                task_guidelines_lines.append(
                    "Spatial statements: use directly observable relations (contact/relative position/containment/alignment/support/open-closed). Each sentence should explicitly name two entities and their visible relation; avoid abstract terms like 'accessible/within reach'."
                )
                task_guidelines_lines.append(
                    "Spatial statements: prefer two-entity relations; if the draft expresses a single-entity state (e.g., open/closed), keep it as-is and do NOT invent new entity names."
                )
                task_guidelines_lines.append(
                    "Spatial statements: format as a single paragraph of complete sentences; each sentence should end with '.', and do NOT add numbering/bullets."
                )
                task_guidelines_lines.append(
                    "Spatial statements: do NOT add, remove, or reorder statements; only copy-edit for grammar/clarity while keeping object names consistent."
                )
            if task_name in (TASK_02, TASK_07):
                task_guidelines_lines.append(
                    "Affordance statements: only include operability/readiness states that are visible or strongly implied by mechanical state (open/closed, sealed/unsealed, empty/full, blocked/unblocked, graspable, stable, separated/clumped). Avoid hidden qualities (sharp/clean/functional heater) unless clearly visible; avoid semantic goals."
                )
                task_guidelines_lines.append(
                    "Affordance statements: format as a single paragraph of complete sentences; each sentence should end with '.', and do NOT add numbering/bullets."
                )
                task_guidelines_lines.append(
                    "Affordance statements: do NOT add, remove, or reorder statements; only copy-edit for grammar/clarity while preserving the affordance/state meaning."
                )
            if task_name == TASK_05:
                task_guidelines_lines.append(
                    "Causal chain (one sentence): prefer the structure 'When <preconditions>, <action/hotspot>; because <mechanism>, <effects>.'"
                )
                task_guidelines_lines.append(
                    "Causal chain (one sentence): MUST cover preconditions, the ongoing interaction, the physical mechanism, and the immediate local effects without adding any new facts."
                )
            if task_name == TASK_13:
                task_guidelines_lines.append(
                    "Rationale: explain why the step is necessary for the overall goal, but do NOT introduce new steps, tools, or objects."
                )
            if task_name == TASK_03:
                task_guidelines_lines.append(
                    "Feasibility (one sentence): keep the feasibility decision and the explicit '(spatial precondition ...)' and '(affordance precondition ...)' status phrases (satisfied/violated/not directly observable); do not drop or alter the status labels."
                )
                task_guidelines_lines.append(
                    "Feasibility: do NOT flip satisfied/violated/not-directly-observable statuses, and do NOT replace the provided precondition clauses with new ones."
                )
            if task_name == TASK_14:
                task_guidelines_lines.append(
                    "Inter-step dependency: explicitly link a previous-step effect to a next-step precondition; keep both step-goal texts unchanged."
                )
            if task_name == TASK_15:
                task_guidelines_lines.append(
                    "Next-step prediction: answer with ONLY the next step goal text verbatim (no paraphrase, no extra explanation)."
                )
            if task_name == TASK_16:
                task_guidelines_lines.append(
                    "Middle-steps infill: preserve the numbering/ordering and quoted step-goal strings exactly; do NOT add/remove/reorder/paraphrase any step goal."
                )
            if task_name == TASK_17:
                task_guidelines_lines.append(
                    "Next-K prediction: preserve the numbering/ordering exactly; do NOT reorder, omit, add, or paraphrase any step goal."
                )
            if task_name == TASK_19:
                task_guidelines_lines.append(
                    "Counterfactual outcome: describe the most likely physical outcome ONLY; do NOT propose recovery actions or advice. "
                    "Avoid any suggestion/advice language (no 'should/need to/must/try/recommend/suggest/consider')."
                )
                task_guidelines_lines.append("Counterfactual outcome: keep it to ONE English sentence.")
            if task_name == TASK_20:
                task_guidelines_lines.append(
                    "Recovery strategy: keep it physically plausible and grounded; do NOT introduce new unseen tools/objects; include a brief spatial+affordance/mechanism justification; avoid generic advice."
                )
                task_guidelines_lines.append(
                    "Recovery strategy: keep the recovery-strategy instruction itself verbatim, then add at most one short justification sentence."
                )
            if task_name == INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK:
                task_guidelines_lines.append(
                    "After-recovery next step: answer with ONLY the next step goal text verbatim (no paraphrase, no extra explanation)."
                )
            task_guidelines = ""
            if task_guidelines_lines:
                task_guidelines = "Task-specific guidelines (follow exactly):\n- " + "\n- ".join(task_guidelines_lines) + "\n"
            constraints = (
                f"{action} the DRAFT_TEXT to improve fluency and logical flow, while preserving ALL factual content.\n"
                "Write in natural, professional QA style (do not just list fields).\n"
                "Prefer minimal edits: keep the draft structure/order when already clear and grounded.\n"
                "If the DRAFT_TEXT already satisfies all requirements, output it unchanged.\n"
                "Do NOT add new objects, actions, states, or claims beyond SOURCE_JSON.\n"
                "Use consistent object naming; do not rename the same object with synonyms.\n"
                "Avoid second-person/imperatives (no 'you should'); answer declaratively.\n"
                "Avoid placeholders like 'unknown', 'N/A', or '...'.\n"
                "Preserve all underscore identifier tokens (tokens containing underscores) EXACTLY as they appear; do not modify them.\n"
                "Do NOT introduce any new underscore identifier tokens.\n"
                "Do NOT mention JSON keys/field names or any prompt block headers; output answer content only.\n"
                "Do NOT include any of these header words in your output: TASK_NAME, SOURCE_JSON, DRAFT_TEXT, REQUIREMENTS, REQUIRED_TOKENS_JSON, ALLOWED_SNAKE_CASE_JSON, OUTPUT.\n"
                "Do NOT use newlines; output as one line unless the task explicitly requires multiple paragraphs.\n"
                "Avoid bullets/list markers; keep any required numbering/quotes exactly as in the draft.\n"
                f"{quote_rule}\n"
                f"{span_rule}\n"
                + task_guidelines
                + (
                    "If the draft uses the phrase \"not directly observable\", keep it when the claim cannot be strictly verified from visible evidence.\n"
                    if must_keep_phrase
                    else ""
                )
                + (
                    "For Task_18, preserve the exact key-value format: FlawStep=...; FlawType=...; Reason=...; Repair: 1) \"...\" 2) \"...\" 3) \"...\".\n"
                    if task_name == TASK_18
                    else ""
                )
                + "Do NOT mention filenames, paths, extensions, timestamps, or frame numbers.\n"
                + "Return ONLY the final answer text.\n"
                + f"{fmt}\n"
            )
            parts = [
                f"TASK_NAME:\n{task_name}",
                "SOURCE_JSON:\n" + json.dumps(fields, ensure_ascii=False),
                "DRAFT_TEXT:\n" + str(draft or ""),
                "REQUIREMENTS:\n" + constraints,
                "REQUIRED_TOKENS_JSON:\n"
                + json.dumps(
                    {"required_snake": required_snake, "required_quotes": required_quotes, "required_facts": required_spans},
                    ensure_ascii=False,
                ),
                "ALLOWED_SNAKE_CASE_JSON:\n" + json.dumps(allowed_snake, ensure_ascii=False),
                "OUTPUT:",
            ]
            return "\n\n".join(parts)

        raw = self.call(system_prompt=SYSTEM_PROMPT, user_text=_prompt(draft_answer, pol=pol0), require_success=bool(require_success))
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            return draft_answer
        out = _defluff_text(_sanitize_answer(task_name, raw))
        if task_name in SINGLE_SENTENCE_TASKS:
            out = _enforce_single_sentence(out)
        if not two_pass:
            return out


        raw2 = self.call(
            system_prompt=SYSTEM_PROMPT,
            user_text=_prompt(out, pol="copyedit"),
            temperature=min(0.2, float(self.cfg.temperature)),
            require_success=bool(require_success),
        )
        raw2 = raw2.strip() if isinstance(raw2, str) else ""
        out2 = _defluff_text(_sanitize_answer(task_name, raw2 or out))
        if task_name in SINGLE_SENTENCE_TASKS:
            out2 = _enforce_single_sentence(out2)
        return out2


def _apply_llm(
    samples: List[Sample],
    llm: TwoStageLlm,
    llm_tasks: set[str],
    *,
    two_pass: bool,
    fallback: str,
    require_success: bool = False,
    vocab_guard: str = "off",
    verify: str = "warn",
) -> List[Sample]:
    if not samples or not llm_tasks or not llm.enabled():
        return samples

    vocab_guard_s = str(vocab_guard or "strict").strip().lower()
    if vocab_guard_s not in {"strict", "warn", "off"}:
        vocab_guard_s = "strict"

    verify_s = str(verify or "strict").strip().lower()
    if verify_s not in {"strict", "warn", "off"}:
        verify_s = "strict"

    out: List[Sample] = []
    attempted = 0
    rewrote = 0
    fell_back = 0
    dropped = 0
    skipped_structured = 0
    vocab_warned = 0
    quote_repaired = 0
    verbatim_used = 0
    for s in samples:
        if s.task_name not in llm_tasks:
            out.append(s)
            continue
        if s.task_name in LLM_SKIP_TASKS:
            skipped_structured += 1
            out.append(s)
            continue
        fields = s.llm_fields or {}

        attempted += 1
        original_answer = s.answer
        new_answer = original_answer
        try:
            new_answer = llm.generate_answer(
                task_name=s.task_name,
                fields=fields,
                draft_answer=original_answer,
                two_pass=two_pass,
                policy="polish",
                require_success=bool(require_success),
            )
        except Exception as e:                    
            logger.warning(f"LLM postprocess failed for task={s.task_name}: {e}")
            if bool(require_success):
                raise
            new_answer = original_answer

        repaired, did = _repair_double_quoted_from_draft(draft=original_answer, candidate=new_answer)
        if did:
            new_answer = repaired
            quote_repaired += 1

        ok, why = _llm_output_is_acceptable(
            task_name=s.task_name,
            draft_answer=original_answer,
            fields=fields,
            candidate=new_answer,
            question=s.question,
            vocab_guard=vocab_guard_s,
            verify=verify_s,
        )
        if not ok:

            retry_policies: List[str] = ["copyedit", "verbatim"]
            retry_ok = False
            last_why = why
            for pol in retry_policies:
                try:
                    cand = llm.generate_answer(
                        task_name=s.task_name,
                        fields=fields,
                        draft_answer=original_answer,
                        two_pass=False,
                        policy=pol,
                        require_success=bool(require_success),
                    )
                except Exception as e:                    
                    logger.warning(f"LLM retry failed for task={s.task_name} policy={pol}: {e}")
                    if bool(require_success):
                        raise
                    continue
                cand2, did2 = _repair_double_quoted_from_draft(draft=original_answer, candidate=cand)
                if did2:
                    cand = cand2
                    quote_repaired += 1
                ok2, why2 = _llm_output_is_acceptable(
                    task_name=s.task_name,
                    draft_answer=original_answer,
                    fields=fields,
                    candidate=cand,
                    question=s.question,
                    vocab_guard=vocab_guard_s,
                    verify=verify_s,
                )
                if ok2:
                    new_answer = cand
                    retry_ok = True
                    ok, why = True, why2
                    if pol == "verbatim":
                        verbatim_used += 1
                    break
                last_why = why2
            if not retry_ok:
                why = last_why
        if not ok:
            if fallback == "fail":
                raise RuntimeError(f"LLM output rejected for task={s.task_name}: {why}")
            if fallback == "skip":
                dropped += 1
                continue
            new_answer = original_answer
            fell_back += 1
        else:
            if _sanitize_space(new_answer) != _sanitize_space(original_answer):
                rewrote += 1
            if vocab_guard_s == "warn" and str(why).startswith("warn:"):
                vocab_warned += 1

        out.append(
            Sample(
                task_name=s.task_name,
                evidence_type=s.evidence_type,
                image=s.image,
                video=s.video,
                question=s.question,
                answer=new_answer,
                source_path=s.source_path,
                llm_fields=s.llm_fields,
            )
        )
    logger.info(
        ">>> [LLM] "
        f"attempted={attempted} rewrote={rewrote} fallback_draft={fell_back} dropped={dropped} "
        f"quote_repaired={quote_repaired} skipped_structured={skipped_structured} "
        f"verbatim_used={verbatim_used} vocab_guard={vocab_guard_s} vocab_warned={vocab_warned} verify={verify_s}"
    )
    return out


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_ASCII_SNAKE_FULL_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _ascii_snake_token(text: Any) -> str:

    s = str(text or "").strip()
    if not s:
        return ""

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()

    s = re.sub(r"[-\\s]+", "_", s)
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


def _fix_final_schema_identifiers_in_plan(plan: Any) -> bool:

    if not isinstance(plan, dict):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return False
    changed = False
    for st in steps:
        if not isinstance(st, dict):
            continue
        cc = st.get("causal_chain")
        if isinstance(cc, dict):
            pat = cc.get("patient")
            if isinstance(pat, str) and pat.strip() and not _ASCII_SNAKE_FULL_RE.fullmatch(pat.strip()):
                new_pat = _ascii_snake_token(pat)
                if new_pat and new_pat != pat:
                    cc["patient"] = new_pat
                    changed = True
        cfs = st.get("critical_frames")
        if not isinstance(cfs, list):
            continue
        for cf in cfs:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if not isinstance(intr, dict):
                continue
            aff = intr.get("affordance_type")
            if isinstance(aff, str) and aff.strip() and not _ASCII_SNAKE_FULL_RE.fullmatch(aff.strip()):
                new_aff = _ascii_snake_token(aff)
                if new_aff and new_aff != aff:
                    intr["affordance_type"] = new_aff
                    changed = True
    return changed


def _validate_final_plan_schema(plan: Dict[str, Any], *, source: str, strict: bool) -> None:
    problems: List[str] = []

    def _add(msg: str) -> None:
        if len(problems) < 80:
            problems.append(msg)

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _check_exact_keys(obj: Any, *, allowed: set[str], path: str) -> None:
        if not strict or not isinstance(obj, dict):
            return
        extra = sorted([k for k in obj.keys() if k not in allowed])
        if extra:
            _add(f"{path} has extra keys: {extra}")
        missing = sorted([k for k in allowed if k not in obj])
        if missing:
            _add(f"{path} missing keys: {missing}")

    def _require_obj(obj: Any, path: str) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            _add(f"{path} must be an object")
            return {}
        return obj

    def _require_list(obj: Any, path: str) -> List[Any]:
        if not isinstance(obj, list):
            _add(f"{path} must be a list")
            return []
        return obj

    def _require_str_field(d: Dict[str, Any], key: str, path: str) -> str:
        if strict and key not in d:
            _add(f"{path}.{key} missing")
            return ""
        v = d.get(key)
        if key in d and not isinstance(v, str):
            _add(f"{path}.{key} must be a string")
            return ""
        s = v.strip() if isinstance(v, str) else ""
        if strict:
            if key in d and not s:
                _add(f"{path}.{key} must be non-empty")
            if s and "\n" in s:
                _add(f"{path}.{key} must not contain newlines")
            if s and _has_frame_leak(s):
                _add(f"{path}.{key} must not reference frame/image indices or filenames")
        return s

    _LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•])\s+")
    _LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)、\)]\s*")

    def _require_causal_list_field(d: Dict[str, Any], key: str, path: str) -> List[str]:
        if strict and key not in d:
            _add(f"{path}.{key} missing")
            return []
        v = d.get(key)
        if key in d and not isinstance(v, list):
            _add(f"{path}.{key} must be a list of strings")
            return []
        out_list: List[str] = []
        if isinstance(v, list):
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    _add(f"{path}.{key}[{i}] must be a string")
                    continue
                s = item.strip()
                if not s:
                    _add(f"{path}.{key}[{i}] must be non-empty")
                    continue
                if "\n" in s:
                    _add(f"{path}.{key}[{i}] must not contain newlines")
                if _LEADING_LIST_MARKER_RE.search(s) or _LEADING_NUMBER_RE.search(s):
                    _add(f"{path}.{key}[{i}] must not start with a list marker or numbering prefix")
                if not s.endswith("."):
                    _add(f"{path}.{key}[{i}] must end with '.'")
                if _STANDALONE_NOT_OBSERVABLE_RE.fullmatch(s):
                    _add(f"{path}.{key}[{i}] must not be a standalone observability marker")
                    continue
                if _has_frame_leak(s):
                    _add(f"{path}.{key}[{i}] must not reference frame/image indices or filenames")
                out_list.append(s)
        if strict and not out_list:
            _add(f"{path}.{key} must be a non-empty list of strings")
        return out_list

    plan_obj = _require_obj(plan, "top")
    _check_exact_keys(plan_obj, allowed={"high_level_goal", "steps"}, path="top")

    _require_str_field(plan_obj, "high_level_goal", "top")

    steps = _require_list(plan_obj.get("steps"), "top.steps")

    allowed_step_keys = {
        "step_id",
        "step_goal",
        "rationale",
        "causal_chain",
        "counterfactual_challenge_question",
        "expected_challenge_outcome",
        "failure_reflecting",
        "critical_frames",
    }
    allowed_step_cc_keys = {
        "agent",
        "action",
        "patient",
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_failure_keys = {"reason", "recovery_strategy"}
    allowed_cf_keys = {"frame_index", "action_state_change_description", "causal_chain", "interaction"}
    allowed_frame_cc_keys = {
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_interaction_keys = {"description", "affordance_type", "mechanism"}

    step_ids: List[int] = []
    for idx, st_any in enumerate(steps):
        path = f"steps[{idx}]"
        if not isinstance(st_any, dict):
            _add(f"{path} must be an object")
            continue
        st = st_any
        _check_exact_keys(st, allowed=allowed_step_keys, path=path)

        sid = st.get("step_id")
        if not _is_int(sid):
            _add(f"{path}.step_id must be int")
        else:
            if int(sid) <= 0:
                _add(f"{path}.step_id must be >= 1")
            step_ids.append(int(sid))

        _require_str_field(st, "step_goal", path)
        rat = _require_str_field(st, "rationale", path)
        q_cf = _require_str_field(st, "counterfactual_challenge_question", path)
        exp = _require_str_field(st, "expected_challenge_outcome", path)
        if strict and q_cf and not re.match(r"^\s*What\s+if\b", q_cf, flags=re.IGNORECASE):
            _add(f"{path}.counterfactual_challenge_question must start with 'What if'")
        cc = _require_obj(st.get("causal_chain"), f"{path}.causal_chain")
        _check_exact_keys(cc, allowed=allowed_step_cc_keys, path=f"{path}.causal_chain")
        for k in ("agent", "action", "patient"):
            v = _require_str_field(cc, k, f"{path}.causal_chain")
            if strict and k == "patient" and v and not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", v):
                _add(
                    f"{path}.causal_chain.patient must be a single snake_case token "
                    f"(got {v!r}; step_id={st.get('step_id')!r})"
                )
        for k in (
            "causal_precondition_on_spatial",
            "causal_precondition_on_affordance",
            "causal_effect_on_spatial",
            "causal_effect_on_affordance",
        ):
            _require_causal_list_field(cc, k, f"{path}.causal_chain")

        fr = _require_obj(st.get("failure_reflecting"), f"{path}.failure_reflecting")
        _check_exact_keys(fr, allowed=allowed_failure_keys, path=f"{path}.failure_reflecting")
        for k in sorted(allowed_failure_keys):
            _require_str_field(fr, k, f"{path}.failure_reflecting")

        cfs = st.get("critical_frames")
        if not isinstance(cfs, list):
            _add(f"{path}.critical_frames must be a list")
            continue
        if strict and len(cfs) != 2:
            _add(f"{path}.critical_frames must have length 2 (got {len(cfs)})")

        fi0: Optional[int] = None
        fi1: Optional[int] = None
        for j, cf_any in enumerate(cfs):
            cf_path = f"{path}.critical_frames[{j}]"
            if not isinstance(cf_any, dict):
                _add(f"{cf_path} must be an object")
                continue
            cf = cf_any
            _check_exact_keys(cf, allowed=allowed_cf_keys, path=cf_path)

            fi = cf.get("frame_index")
            if not _is_int(fi):
                _add(f"{cf_path}.frame_index must be int")
            else:
                if int(fi) <= 0:
                    _add(f"{cf_path}.frame_index must be >= 1")
                if j == 0:
                    fi0 = int(fi)
                elif j == 1:
                    fi1 = int(fi)

            _require_str_field(cf, "action_state_change_description", cf_path)

            fcc = _require_obj(cf.get("causal_chain"), f"{cf_path}.causal_chain")
            _check_exact_keys(fcc, allowed=allowed_frame_cc_keys, path=f"{cf_path}.causal_chain")
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                _require_causal_list_field(fcc, k, f"{cf_path}.causal_chain")

            intr = _require_obj(cf.get("interaction"), f"{cf_path}.interaction")
            _check_exact_keys(intr, allowed=allowed_interaction_keys, path=f"{cf_path}.interaction")
            for k in sorted(allowed_interaction_keys):
                v = _require_str_field(intr, k, f"{cf_path}.interaction")
                if strict and k == "affordance_type" and v and not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", v):
                    _add(
                        f"{cf_path}.interaction.affordance_type must be a single snake_case token "
                        f"(got {v!r}; e.g., pressable_surface)"
                    )

        if strict and fi0 is not None and fi1 is not None:
            if fi0 == fi1:
                _add(f"{path}.critical_frames frame_index must be distinct (got {fi0} and {fi1})")
            if fi0 > fi1:
                _add(f"{path}.critical_frames must be in increasing time order (frame_index {fi0} then {fi1})")

    if strict and step_ids:
        dup = sorted({sid for sid in step_ids if step_ids.count(sid) > 1})
        if dup:
            _add(f"steps.step_id must be unique (duplicates: {dup})")

    if problems:
        raise ValueError(f"Final schema validation failed: source={source} problems=" + " | ".join(problems[:10]))


def _has_frame_leak(text: str) -> bool:
    s = str(text or "")
    for pat in FRAME_LEAK_PATTERNS:
        if pat.search(s):
            return True
    return False


def _sanitize_space(text: str) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_key_moment_prefix(text: str) -> str:
    s = str(text or "").strip()
    s = _KEY_MOMENT_PREFIX_RE.sub("", s)
    return s.strip()


def _split_numbered_block(value: Any) -> List[str]:

    if isinstance(value, list):
        out_list: List[str] = []
        for item in value:
            out_list.extend(_split_numbered_block(item))
        return [x for x in out_list if x]

    raw = str(value or "").replace("\\n", "\n").strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:[-*•])\s*", "", ln).strip()
        ln = re.sub(r"^\s*\d+\s*[\.\)、\)]\s*", "", ln).strip()
        if _STANDALONE_NOT_OBSERVABLE_RE.fullmatch(ln):
            continue
        if ln:
            out.append(ln)
    if not out and raw:
        out = [raw]
    return out


def _pick_first_points(value: Any, max_points: int) -> str:
    pts = _split_numbered_block(value)
    if not pts:
        return ""
    return " ".join(pts[: max(1, int(max_points))]).strip()


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _parse_spatial_relations(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            out.append(item)
    return out


def _parse_affordance_states(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            out.append(item)
    return out


def _relation_phrase(token: str) -> str:
    return str(token or "").strip().replace("_", " ")

def _coerce_truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0", ""):
            return False
    return True


def _format_spatial_relation(rel: Dict[str, Any]) -> str:
    relation = _relation_phrase(rel.get("relation", ""))
    if not relation:
        return ""
    objects = rel.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    objects = [str(o).strip() for o in objects if isinstance(o, (str, int, float)) and str(o).strip()]
    truth_bool = _coerce_truth(rel.get("truth", True))

    if not objects:
        base = f"the relevant objects are {relation}"
    elif len(objects) == 1:
        base = f"{objects[0]} is {relation}"
    else:
        base = f"{objects[0]} is {relation} " + " and ".join(objects[1:])

    if not truth_bool:
        base = base.replace(" is ", " is not ", 1)
    return base.strip()


def _format_affordance_state(st: Dict[str, Any]) -> str:
    obj = str(st.get("object_name", "") or "").strip()
    affs = st.get("affordance_types", [])
    if not isinstance(affs, list):
        affs = []
    affs = [str(a).strip() for a in affs if isinstance(a, (str, int, float)) and str(a).strip()]
    reasons = str(st.get("reasons", "") or "").strip()
    if not obj and not affs:
        return ""
    if obj and affs:
        base = f"{obj} has affordance/state " + ", ".join(affs)
    elif obj:
        base = f"{obj} has the required affordance/state"
    else:
        base = "the object has affordance/state " + ", ".join(affs)
    if reasons:
        base = f"{base} because {reasons}"
    return base.strip()


def _format_affordance_state_compact(st: Dict[str, Any]) -> str:
    obj = str(st.get("object_name", "") or "").strip()
    affs = st.get("affordance_types", [])
    if not isinstance(affs, list):
        affs = []
    affs = [str(a).strip() for a in affs if isinstance(a, (str, int, float)) and str(a).strip()]
    if obj and affs:
        return f"{obj} " + ", ".join(affs)
    if obj:
        return obj
    return ", ".join(affs)


def _format_spatial(value: Any, *, max_items: int) -> str:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _pick_first_points(value, max_items)
    rels = _parse_spatial_relations(value)
    phrases = [_format_spatial_relation(r) for r in rels]
    phrases = [p for p in phrases if p]
    return " ".join(phrases[: max(1, int(max_items))]).strip()


def _format_affordance(value: Any, *, max_items: int) -> str:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _pick_first_points(value, max_items)
    sts = _parse_affordance_states(value)
    phrases = [_format_affordance_state(s) for s in sts]
    phrases = [p for p in phrases if p]
    return " ".join(phrases[: max(1, int(max_items))]).strip()

def _format_spatial_points(value: Any) -> List[str]:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _split_numbered_block(value)
    rels = _parse_spatial_relations(value)
    pts = [_format_spatial_relation(r) for r in rels]
    return [p for p in pts if p]


def _format_affordance_points(value: Any) -> List[str]:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _split_numbered_block(value)
    sts = _parse_affordance_states(value)
    pts = [_format_affordance_state(s) for s in sts]
    return [p for p in pts if p]


def _terms_from_spatial(value: Any) -> set[str]:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _normalize_terms(" ".join(_split_numbered_block(value)))
    out: set[str] = set()
    for rel in _parse_spatial_relations(value):
        rel_token = rel.get("relation")
        if isinstance(rel_token, str) and rel_token.strip():
            out.add(rel_token.strip().lower())
        objs = rel.get("objects", [])
        if isinstance(objs, list):
            for o in objs:
                if isinstance(o, str) and o.strip():
                    out.add(o.strip().lower())
    out -= _GENERIC_OBJECT_TOKENS
    return out


def _terms_from_affordance(value: Any) -> set[str]:
    if isinstance(value, str) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
        return _normalize_terms(" ".join(_split_numbered_block(value)))
    out: set[str] = set()
    for st in _parse_affordance_states(value):
        obj = st.get("object_name")
        if isinstance(obj, str) and obj.strip():
            out.add(obj.strip().lower())
        affs = st.get("affordance_types", [])
        if isinstance(affs, list):
            for a in affs:
                if isinstance(a, str) and a.strip():
                    out.add(a.strip().lower())
    out -= _GENERIC_OBJECT_TOKENS
    return out

def _parse_timestamp_from_path(path: str) -> Optional[float]:
    m = _TS_RE.search(os.path.basename(path or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_relpath(path: str, root: str) -> str:
    ap = os.path.abspath(path)
    ar = os.path.abspath(root)
    try:
        rel = os.path.relpath(ap, ar)
        return rel.replace("\\", "/")
    except Exception:
        return ap.replace("\\", "/")


def _list_item_dirs(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "causal_plan_with_keyframes.json" in filenames:
            out.append(dirpath)
            dirnames[:] = []
    return sorted(out)


def _list_sampled_frames(item_dir: str) -> List[str]:
    dirs = [
        os.path.join(item_dir, "sampled_frames"),
        os.path.join(item_dir, "stage1", "sampled_frames"),
    ]
    patterns = [
        "sample_*.jpg",
        "sample_*.jpeg",
        "sample_*.png",
    ]
    for d in dirs:
        paths: List[str] = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(d, pat)))
        if paths:
            return sorted(paths)
    return []


def _pick_uniform(frames: Sequence[str], k: int) -> List[str]:
    n = len(frames)
    if n == 0:
        return []
    k = max(1, int(k))
    if n <= k:
        return list(frames)
    if k == 1:
        return [frames[n // 2]]
    idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    uniq = []
    for i in idxs:
        if not uniq or uniq[-1] != i:
            uniq.append(i)
    return [frames[i] for i in uniq]


def _pick_head_tail(frames: Sequence[str], head: int, tail: int) -> List[str]:
    head = max(0, int(head))
    tail = max(0, int(tail))
    if not frames:
        return []
    if head + tail <= 0:
        return []
    if len(frames) <= head + tail:
        return list(frames)
    return list(frames[:head]) + list(frames[-tail:])


def _resolve_video_prefix(item_dir: str, step_id: int) -> Optional[str]:
    cands = [
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}.mp4"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _resolve_video_clip(item_dir: str, step_id: int) -> Optional[str]:
    if step_id <= 0:
        return None
    cands: List[str] = []
    if step_id == 1:
        cands.append(os.path.join(item_dir, "last_frame_segments", "segment_start_to_step01.mp4"))
        cands.append(os.path.join(item_dir, "last_frame_segments", "segment_start_to_step1.mp4"))
    else:
        cands.append(os.path.join(item_dir, "last_frame_segments", f"segment_step{step_id - 1:02d}_to_step{step_id:02d}.mp4"))
        cands.append(os.path.join(item_dir, "last_frame_segments", f"segment_step{step_id - 1}_to_step{step_id}.mp4"))
    for p in cands:
        if os.path.exists(p):
            return p


    seg_path = os.path.join(item_dir, "stage2", "step_segments.json")
    if os.path.exists(seg_path):
        try:
            seg_json = _read_json(seg_path)
            segments = seg_json.get("segments", [])
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    sid = seg.get("step_id")
                    try:
                        sid_int = int(sid)
                    except Exception:
                        continue
                    if sid_int != int(step_id):
                        continue
                    clip_rel = seg.get("clip_relpath")
                    if isinstance(clip_rel, str) and clip_rel.strip():
                        cand = os.path.join(item_dir, "stage2", clip_rel.strip())
                        if os.path.exists(cand):
                            return cand
        except Exception:
            pass

    clips_dir = os.path.join(item_dir, "stage2", "step_clips")
    if os.path.isdir(clips_dir):
        matches = sorted(glob.glob(os.path.join(clips_dir, f"step{step_id:02d}_*.mp4")))
        if matches:
            return matches[0]
    return None


def _resolve_step_clip_start_sec(item_dir: str, step_id: int) -> Optional[float]:

    for step_dir in sorted(glob.glob(os.path.join(item_dir, f"{int(step_id):02d}_*"))):
        meta_path = os.path.join(step_dir, "step_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = _read_json(meta_path)
            v = meta.get("clip_start_sec")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            if isinstance(v, str) and v.strip():
                return float(v.strip())
        except Exception:
            continue


    seg_path = os.path.join(item_dir, "stage2", "step_segments.json")
    if os.path.exists(seg_path):
        try:
            seg_json = _read_json(seg_path)
            segments = seg_json.get("segments", [])
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        sid = int(seg.get("step_id", 0) or 0)
                    except Exception:
                        continue
                    if sid != int(step_id):
                        continue
                    v = seg.get("start_sec")
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
                    if isinstance(v, str) and v.strip():
                        return float(v.strip())
        except Exception:
            pass


    for step_dir in sorted(glob.glob(os.path.join(item_dir, f"{int(step_id):02d}_*"))):
        man_path = os.path.join(step_dir, "frame_manifest.json")
        if not os.path.exists(man_path):
            continue
        try:
            man = _read_json(man_path)
            frames = man.get("frames", [])
            if not isinstance(frames, list):
                continue
            ts_vals: List[float] = []
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                v = fr.get("timestamp_sec")
                try:
                    ts_vals.append(float(v))
                except Exception:
                    continue
            if ts_vals:
                return float(min(ts_vals))
        except Exception:
            continue
    return None


def _resolve_frame_timestamp_sec(item_dir: str, step_id: int, frame_index_1based: int) -> Optional[float]:
    if int(step_id) <= 0 or int(frame_index_1based) <= 0:
        return None
    key = (os.path.abspath(item_dir), int(step_id))
    if key not in _FRAME_TS_CACHE:
        ts_map: Dict[int, float] = {}
        step_globs = [
            os.path.join(item_dir, f"{int(step_id):02d}_*"),
            os.path.join(item_dir, f"{int(step_id)}_*"),
        ]
        for g in step_globs:
            for step_dir in sorted(glob.glob(g)):
                man_path = os.path.join(step_dir, "frame_manifest.json")
                if not os.path.exists(man_path):
                    continue
                try:
                    man = _read_json(man_path)
                    frames = man.get("frames", [])
                    if not isinstance(frames, list):
                        continue
                    for fr in frames:
                        if not isinstance(fr, dict):
                            continue
                        fi = fr.get("frame_index_1based")
                        ts = fr.get("timestamp_sec")
                        try:
                            fi_int = int(fi)
                            ts_f = float(ts)
                        except Exception:
                            continue
                        if fi_int > 0:
                            ts_map[fi_int] = ts_f
                    if ts_map:
                        break
                except Exception:
                    continue
            if ts_map:
                break
        _FRAME_TS_CACHE[key] = ts_map
    return _FRAME_TS_CACHE.get(key, {}).get(int(frame_index_1based))


def _task15_16_prefix_clip_path(item_dir: str, *, step_id: int, critical_frame_index: int, end_sec: float) -> str:
    end_ms = int(round(float(end_sec) * 1000.0))
    return os.path.join(
        item_dir,
        "critical_frame_pre3s_segments",
        f"step{int(step_id):02d}_cf{int(critical_frame_index)}_end_{end_ms}ms.mp4",
    )


def _ffmpeg_exists(ffmpeg_bin: str) -> bool:
    s = str(ffmpeg_bin or "").strip()
    if not s:
        return False

    if os.path.isabs(s) or os.path.sep in s or (os.path.altsep and os.path.altsep in s):
        return os.path.exists(s) and os.access(s, os.X_OK)
    return shutil.which(s) is not None


def _ffmpeg_cut_prefix(
    *,
    ffmpeg_bin: str,
    src: str,
    duration_sec: float,
    dst: str,
    overwrite: bool,
) -> bool:
    if duration_sec <= 0.05:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "0",
        "-i",
        src,
        "-t",
        f"{float(duration_sec):.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        dst,
    ]
    cmd.insert(1, "-y" if overwrite else "-n")
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:

        cmd2 = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0",
            "-i",
            src,
            "-t",
            f"{float(duration_sec):.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            dst,
        ]
        cmd2.insert(1, "-y" if overwrite else "-n")
        try:
            subprocess.run(cmd2, check=True)
            return True
        except Exception:
            return False


def _ffmpeg_concat_videos(
    *,
    ffmpeg_bin: str,
    srcs: Sequence[str],
    dst: str,
    overwrite: bool,
) -> bool:
    if not srcs:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    list_path = dst + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for s in srcs:
                ap = os.path.abspath(str(s))
                ap = ap.replace("\\", "/")
                ap = ap.replace("'", "\\'")
                f.write(f"file '{ap}'\n")
    except Exception:
        return False

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        dst,
    ]
    cmd.insert(1, "-y" if overwrite else "-n")
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:

        cmd2 = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            dst,
        ]
        cmd2.insert(1, "-y" if overwrite else "-n")
        try:
            subprocess.run(cmd2, check=True)
            return True
        except Exception:
            return False
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


def _ensure_video_prefix(
    item_dir: str,
    step_id: int,
    *,
    ffmpeg_bin: str,
    build: bool,
    overwrite: bool,
) -> Optional[str]:
    existing = _resolve_video_prefix(item_dir, step_id)
    if existing:
        return existing
    if not build:
        return None

    dst = os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{int(step_id):02d}_last.mp4")
    if os.path.exists(dst) and not overwrite:
        return dst

    clips: List[str] = []
    for sid in range(1, int(step_id) + 1):
        clip = _resolve_video_clip(item_dir, sid)
        if not clip:
            return None
        clips.append(clip)
    ok = _ffmpeg_concat_videos(ffmpeg_bin=ffmpeg_bin, srcs=clips, dst=dst, overwrite=overwrite)
    if ok and os.path.exists(dst):
        return dst
    return None


def _resolve_video_prefix_or_step_clip(
    item_dir: str,
    prefix_end_step: int,
    *,
    ffmpeg_bin: str,
    build: bool,
    overwrite: bool,
    strict: bool,
) -> Optional[str]:

    if int(prefix_end_step) <= 0:
        return None
    vid = _ensure_video_prefix(
        item_dir,
        int(prefix_end_step),
        ffmpeg_bin=ffmpeg_bin,
        build=build,
        overwrite=overwrite,
    )
    if vid:
        return vid
    if strict:
        return None
    return _resolve_video_clip(item_dir, int(prefix_end_step))


def _find_keyframe_image(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:
    step_globs = [
        os.path.join(item_dir, f"{int(step_id):02d}_*"),
        os.path.join(item_dir, f"{int(step_id)}_*"),
    ]
    seen = set()
    pats: List[str] = []
    for step_prefix in step_globs:
        if step_prefix in seen:
            continue
        seen.add(step_prefix)
        for ext in ("jpg", "jpeg", "png"):
            pats.append(os.path.join(step_prefix, f"frame_{int(frame_index):03d}_ts_*.{ext}"))
            pats.append(os.path.join(step_prefix, f"frame_{int(frame_index):03d}_*.{ext}"))
    for pat in pats:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def _extract_snake_case_objects(plan: Dict[str, Any]) -> List[str]:
    tokens: set[str] = set()
    for step in plan.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        cc = step.get("causal_chain") or {}
        if isinstance(cc, dict):
            for k in ("agent", "patient"):
                v = cc.get(k)
                if isinstance(v, str):
                    tokens |= set(_SNAKE_TOKEN_RE.findall(v))
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                v = cc.get(k)
                if isinstance(v, str):
                    tokens |= set(_SNAKE_TOKEN_RE.findall(v))
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    for x in v:
                        tokens |= set(_SNAKE_TOKEN_RE.findall(x))
        for cf in step.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction") or {}
            if isinstance(intr, dict):
                for k in ("description", "affordance_type", "mechanism"):
                    v = intr.get(k)
                    if isinstance(v, str):
                        tokens |= set(_SNAKE_TOKEN_RE.findall(v))
            fcc = cf.get("causal_chain") or {}
            if isinstance(fcc, dict):
                for k in (
                    "causal_precondition_on_spatial",
                    "causal_precondition_on_affordance",
                    "causal_effect_on_spatial",
                    "causal_effect_on_affordance",
                ):
                    v = fcc.get(k)
                    if isinstance(v, str):
                        tokens |= set(_SNAKE_TOKEN_RE.findall(v))
                    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                        for x in v:
                            tokens |= set(_SNAKE_TOKEN_RE.findall(x))
    cleaned = [t for t in tokens if t not in _GENERIC_OBJECT_TOKENS]
    cleaned.sort()
    return cleaned


def _extract_key_objects_for_task02(plan: Dict[str, Any]) -> List[str]:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return []


    is_structured_schema = False
    for st in steps:
        cc = st.get("causal_chain")
        if isinstance(cc, dict):
            sp = cc.get("causal_precondition_on_spatial")
            af = cc.get("causal_precondition_on_affordance")
            if isinstance(sp, list) and any(isinstance(x, dict) for x in sp):
                is_structured_schema = True
                break
            if isinstance(af, list) and any(isinstance(x, dict) for x in af):
                is_structured_schema = True
                break
        for cf in st.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict):
                is_structured_schema = True
                break
        if is_structured_schema:
            break

    if is_structured_schema:
        objs: set[str] = set()

        def _add_obj(x: Any) -> None:
            if not isinstance(x, str):
                return
            t = x.strip()
            if not t:
                return
            if t.lower() in _GENERIC_OBJECT_TOKENS:
                return
            objs.add(t)

        def _add_from_spatial(val: Any) -> None:
            for rel in _parse_spatial_relations(val):
                for o in rel.get("objects", []) if isinstance(rel.get("objects", []), list) else []:
                    _add_obj(str(o))

        def _add_from_affordance(val: Any) -> None:
            for stt in _parse_affordance_states(val):
                _add_obj(stt.get("object_name"))

        for st in steps:
            cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
            if isinstance(cc, dict):
                _add_obj(cc.get("patient"))
                _add_from_spatial(cc.get("causal_precondition_on_spatial"))
                _add_from_spatial(cc.get("causal_effect_on_spatial"))
                _add_from_affordance(cc.get("causal_precondition_on_affordance"))
                _add_from_affordance(cc.get("causal_effect_on_affordance"))

            for cf in st.get("critical_frames", []) or []:
                if not isinstance(cf, dict):
                    continue
                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                if isinstance(fcc, dict):
                    _add_obj(fcc.get("patient"))
                    _add_from_spatial(fcc.get("causal_precondition_on_spatial"))
                    _add_from_spatial(fcc.get("causal_effect_on_spatial"))
                    _add_from_affordance(fcc.get("causal_precondition_on_affordance"))
                    _add_from_affordance(fcc.get("causal_effect_on_affordance"))

                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                if isinstance(intr, dict):
                    for t in intr.get("tools", []) if isinstance(intr.get("tools", []), list) else []:
                        _add_obj(str(t))
                    for m in intr.get("materials", []) if isinstance(intr.get("materials", []), list) else []:
                        _add_obj(str(m))

        tokens = sorted(objs)
        if len(tokens) > 12:
            tokens = tokens[:12]
        return tokens




    patient_pool: set[str] = set()
    patient_pool_lower: set[str] = set()
    for st in steps:
        cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        if not isinstance(cc, dict):
            continue
        pat = cc.get("patient")
        if isinstance(pat, str) and pat.strip():
            t = pat.strip()
            if t.lower() not in _GENERIC_OBJECT_TOKENS:
                patient_pool.add(t)
                patient_pool_lower.add(t.lower())

    agent_pool_lower: set[str] = set()
    for st in steps:
        cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        if not isinstance(cc, dict):
            continue
        ag = cc.get("agent")
        if isinstance(ag, str) and ag.strip():
            agent_pool_lower.add(ag.strip().lower())

    affordance_types: set[str] = set()
    for st in steps:
        for cf in st.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if isinstance(intr, dict):
                v = intr.get("affordance_type")
                if isinstance(v, str) and v.strip():
                    affordance_types.add(v.strip().lower())

    objs = set(patient_pool) | set(_extract_snake_case_objects(plan))

    def _is_object_like(token: str) -> bool:
        t = (token or "").strip()
        if not t:
            return False
        tl = t.lower()

        if tl in agent_pool_lower and tl not in patient_pool_lower:
            return False
        if tl in _STOPWORDS:
            return False
        if tl in _GENERIC_OBJECT_TOKENS or tl in _VERBISH_TOKENS:
            return False
        if tl in affordance_types:
            return False
        if tl in _RELATION_TOKENS or tl in _GENERIC_PART_TOKENS:
            return False
        if tl.startswith(("ready_to_", "partially_", "more_", "less_", "switched_")):
            return False
        if tl.endswith(("_on", "_off", "_open", "_closed", "_pressed", "_depressed")):
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t):
            return False

        if "_" not in t and tl not in patient_pool_lower:
            return False
        return True

    tokens = [t for t in objs if _is_object_like(t)]
    tokens = sorted(set(tokens))
    if len(tokens) > 12:
        tokens = tokens[:12]
    return tokens


def _stable_int_seed(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _normalize_terms(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "be",
        "by",
        "for",
        "with",
        "then",
        "into",
        "from",
        "that",
        "this",
        "it",
        "as",
        "after",
        "before",
        "when",
        "while",
    }
    out = {t for t in tokens if t not in stop and len(t) >= 3}
    out -= _GENERIC_OBJECT_TOKENS
    return out


def _has_dependency(prev_effects: Any, next_preconds: Any) -> bool:
    eff = _terms_from_spatial(prev_effects) | _terms_from_affordance(prev_effects)
    pre = _terms_from_spatial(next_preconds) | _terms_from_affordance(next_preconds)
    if not eff or not pre:
        return False
    if eff & pre:
        return True
    for e in eff:
        for p in pre:
            if e in p or p in e:
                return True
    return False


def _sharegpt_entry(sample: Sample, *, attach_evidence: bool) -> Dict[str, Any]:
    source_path_norm = str(sample.source_path or "").replace("\\", "/").strip()
    item_dir = ""
    if source_path_norm.endswith("/causal_plan_with_keyframes.json"):
        item_dir = source_path_norm[: -len("/causal_plan_with_keyframes.json")].rstrip("/")
    elif "/" in source_path_norm:
        item_dir = source_path_norm.rsplit("/", 1)[0].rstrip("/")

    entry: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "image": list(sample.image) if bool(attach_evidence) else [],
        "conversations": [
            {"from": "human", "value": sample.question},
            {"from": "gpt", "value": sample.answer},
        ],
        "meta": {
            "task_name": sample.task_name,
            "evidence_type": sample.evidence_type,
            "source_path": sample.source_path,
            **({"item_dir": item_dir} if item_dir else {}),
            **({"evidence_files": list(sample.image) + ([sample.video] if sample.video else [])} if bool(attach_evidence) else {}),
        },
    }
    if bool(attach_evidence) and sample.video:
        entry["video"] = sample.video
    return entry


def _write_jsonl(out_path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_RUN_CONFIG_BASENAME = "run_config.json"
_RESUME_STATE_BASENAME = "resume_state.jsonl"
_TASKSLIST_RUN_CONFIG_ID = "task_list_run_config"


def _infer_item_dir_from_source_path(source_path: str) -> str:
    src = str(source_path or "").replace("\\", "/").strip()
    if not src:
        return ""
    if src.endswith("/causal_plan_with_keyframes.json"):
        return src[: -len("/causal_plan_with_keyframes.json")].rstrip("/")
    if "/" in src:
        return src.rsplit("/", 1)[0].rstrip("/")
    return src


def _run_config_path(output_dir: str) -> str:
    return os.path.join(output_dir, _RUN_CONFIG_BASENAME)


def _resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, _RESUME_STATE_BASENAME)


def _load_json_maybe(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load json: {path} ({type(e).__name__}: {e})")
        return None


def _write_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _issues_paths(*, output_dir: str, prefix: str) -> Dict[str, str]:
    p = str(prefix or "issues").strip()
    if not p:
        p = "issues"
    base = os.path.join(output_dir, p)
    return {
        "errors_jsonl": base + "_errors.jsonl",
        "warnings_jsonl": base + "_warnings.jsonl",
        "errors_json": base + "_errors.json",
        "warnings_json": base + "_warnings.json",
        "summary_json": base + "_summary.json",
    }


def _log_issue(
    *,
    output_dir: str,
    prefix: str,
    severity: str,
    phase: str,
    message: str,
    rel_item: str = "",
    task_name: str = "",
    sample_id: str = "",
    details: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> None:
    sev = str(severity or "").strip().lower()
    if sev not in {"error", "warn", "warning"}:
        sev = "warn"
    sev_norm = "error" if sev == "error" else "warn"
    paths = _issues_paths(output_dir=output_dir, prefix=prefix)
    path = paths["errors_jsonl"] if sev_norm == "error" else paths["warnings_jsonl"]
    rec: Dict[str, Any] = {
        "severity": sev_norm,
        "phase": str(phase or "").strip(),
        "message": str(message or "").strip(),
    }
    if rel_item:
        rec["rel_item"] = str(rel_item)
    if task_name:
        rec["task_name"] = str(task_name)
    if sample_id:
        rec["sample_id"] = str(sample_id)
    if details:
        rec["details"] = details
    if exc is not None:
        rec["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            rec["traceback"] = traceback.format_exc(limit=20)
        except Exception:
            pass
    try:
        _append_jsonl(path, rec)
    except Exception as e:                    
        logger.warning(f"Failed to write issue log ({type(e).__name__}: {e})")


def _finalize_issue_json_files(*, output_dir: str, prefix: str) -> None:
    paths = _issues_paths(output_dir=output_dir, prefix=prefix)

    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = str(line or "").strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
        except Exception:
            return out
        return out

    errors = _read_jsonl(paths["errors_jsonl"])
    warns = _read_jsonl(paths["warnings_jsonl"])
    try:
        _write_json_atomic(paths["errors_json"], errors)
    except Exception:
        pass
    try:
        _write_json_atomic(paths["warnings_json"], warns)
    except Exception:
        pass
    try:
        _write_json_atomic(
            paths["summary_json"],
            {
                "errors": int(len(errors)),
                "warnings": int(len(warns)),
                "errors_jsonl": paths["errors_jsonl"],
                "warnings_jsonl": paths["warnings_jsonl"],
                "errors_json": paths["errors_json"],
                "warnings_json": paths["warnings_json"],
            },
        )
    except Exception:
        pass


def _load_resume_state(state_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(state_path):
        return out
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                rel_item = rec.get("rel_item")
                if not isinstance(rel_item, str) or not rel_item.strip():
                    continue
                out[rel_item.strip()] = rec
    except Exception as e:
        logger.warning(f"Failed to read resume state: {state_path} ({type(e).__name__}: {e})")
        return {}
    return out


def _infer_processed_items_from_output_dir(output_dir: str, enabled_tasks: Sequence[str]) -> set[str]:
    processed: set[str] = set()

    for task in sorted(set(enabled_tasks or [])):
        jsonl_path = os.path.join(output_dir, task, "data.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        entry = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                    item_dir = meta.get("item_dir")
                    if isinstance(item_dir, str) and item_dir.strip():
                        processed.add(item_dir.strip())
                        continue
                    src = meta.get("source_path") or ""
                    if isinstance(src, str) and src.strip():
                        inferred = _infer_item_dir_from_source_path(src)
                        if inferred:
                            processed.add(inferred)
        except Exception:
            continue

    return processed


def _jsonl_has_any_entry(jsonl_path: str) -> bool:
    if not os.path.exists(jsonl_path):
        return False
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def _build_taskslist_run_config(
    *,
    input_root: str,
    enabled_tasks: Sequence[str],
    text_only: bool,
    meta_abs_paths: bool,
    uniform_k: int,
    head: int,
    tail: int,
    require_videos: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
    strict_schema: bool,
    min_steps: int,
    llm_enabled: bool,
    llm_tasks: Sequence[str],
    llm_max_tokens: int,
    llm_temperature: float,
    llm_single_pass: bool,
    llm_vocab_guard: str,
    llm_verify: str,
    llm_fallback: str,
    llm_require_success: bool,
) -> Dict[str, Any]:
    return {
        "config_id": _TASKSLIST_RUN_CONFIG_ID,
        "generator": "generate_stage_one_qa",
        "input_root": os.path.abspath(input_root),
        "enabled_tasks": sorted(set(enabled_tasks or [])),
        "text_only": bool(text_only),
        "meta_abs_paths": bool(meta_abs_paths),
        "uniform_k": int(uniform_k),
        "head": int(head),
        "tail": int(tail),
        "require_videos": bool(require_videos),
        "ffmpeg_bin": str(ffmpeg_bin or ""),
        "build_video_prefix_clips": bool(build_video_prefix_clips),
        "overwrite_video_prefix_clips": bool(overwrite_video_prefix_clips),
        "strict_prefix_video": bool(strict_prefix_video),
        "strict_schema": bool(strict_schema),
        "min_steps": int(min_steps),
        "llm_enabled": bool(llm_enabled),
        "llm_tasks": sorted(set(llm_tasks or [])),
        "llm_max_tokens": int(llm_max_tokens),
        "llm_temperature": float(llm_temperature),
        "llm_single_pass": bool(llm_single_pass),
        "llm_vocab_guard": str(llm_vocab_guard or "").strip().lower(),
        "llm_verify": str(llm_verify or "").strip().lower(),
        "llm_fallback": str(llm_fallback or ""),
        "llm_require_success": bool(llm_require_success),
    }


def _run_config_mismatch_keys(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    keys = sorted(set(new.keys()))
    mism: List[str] = []
    for k in keys:
        old_v = old.get(k)
        new_v = new.get(k)

        if k == "llm_vocab_guard" and (old_v is None or old_v == ""):
            old_v = "strict"

        if k == "llm_verify" and (old_v is None or old_v == ""):
            old_v = "strict"
        if old_v != new_v:
            mism.append(k)
    return mism


@dataclass(frozen=True)
class AuditIssue:
    severity: str                    
    task_name: str
    file: str
    line: int
    sample_id: str
    message: str


@dataclass(frozen=True)
class AuditReport:
    total_samples: int
    errors: int
    warnings: int
    issues: List[AuditIssue]


def _allowed_evidence_types_for_task(task_name: str) -> set[str]:
    mapping: Dict[str, set[str]] = {
        TASK_08: {EVIDENCE_PREFIX, EVIDENCE_UNIFORM},
        TASK_09: {EVIDENCE_UNIFORM},
        TASK_10: {EVIDENCE_CLIP, EVIDENCE_KEYFRAME},
        INTERNAL_PATIENT_IDENTIFICATION_TASK: {EVIDENCE_KEYFRAME},
        TASK_11: {EVIDENCE_KEYFRAME},
        INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK: {EVIDENCE_KEYFRAME},
        INTERNAL_HOTSPOT_MECHANISM_TASK: {EVIDENCE_KEYFRAME},
        TASK_04: {EVIDENCE_KEYFRAME},
        TASK_12: {EVIDENCE_KEYFRAME},
        TASK_05: {EVIDENCE_KEYFRAME},
        TASK_13: {EVIDENCE_KEYFRAME},
        TASK_01: {EVIDENCE_KEYFRAME},
        TASK_02: {EVIDENCE_KEYFRAME},
        TASK_03: {EVIDENCE_KEYFRAME},
        TASK_06: {EVIDENCE_KEYFRAME},
        TASK_07: {EVIDENCE_KEYFRAME},
        TASK_14: {EVIDENCE_KEYFRAME},
        TASK_15: {EVIDENCE_PREFIX},
        TASK_16: {EVIDENCE_UNIFORM},
        TASK_17: {EVIDENCE_PREFIX},
        TASK_18: {EVIDENCE_PREFIX},
        TASK_19: {EVIDENCE_KEYFRAME},
        TASK_20: {EVIDENCE_KEYFRAME},
        INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK: {EVIDENCE_PREFIX},
    }
    return set(mapping.get(task_name, set()))


def _abs_under_root(path: str, root: str) -> str:
    p = str(path or "").strip()
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(root, p))


def _audit_output_dir(
    *,
    output_dir: str,
    input_root: str,
    max_issues: int,
    require_evidence: bool = True,
    issue_sink: Optional[Any] = None,
) -> AuditReport:
    issues: List[AuditIssue] = []
    error_count = 0
    warn_count = 0
    total_samples = 0

    def _add(sev: str, task: str, file: str, line: int, sid: str, msg: str) -> None:
        nonlocal error_count, warn_count
        if str(sev) == "error":
            error_count += 1
        else:
            warn_count += 1
        iss = AuditIssue(severity=sev, task_name=task, file=file, line=int(line), sample_id=sid, message=msg)
        if issue_sink is not None:
            try:
                issue_sink(iss)
            except Exception:
                pass
        if len(issues) < max(1, int(max_issues)):
            issues.append(iss)

    jsonl_paths = sorted(glob.glob(os.path.join(output_dir, "*", "data.jsonl")))
    if not jsonl_paths:
        _add("error", "", output_dir, 0, "", "No task data.jsonl found under output_dir.")
        return AuditReport(total_samples=0, errors=error_count, warnings=warn_count, issues=issues)

    for jsonl_path in jsonl_paths:
        expected_task = os.path.basename(os.path.dirname(jsonl_path))
        try:
            f = open(jsonl_path, "r", encoding="utf-8")
            lines = f
        except Exception as e:
            _add("error", expected_task, jsonl_path, 0, "", f"Failed to read jsonl: {e}")
            continue

        for ln_no, line in enumerate(lines, start=1):
            raw = str(line or "").strip()
            if not raw:
                _add("warn", expected_task, jsonl_path, ln_no, "", "Empty line in jsonl.")
                continue
            try:
                entry = json.loads(raw)
            except Exception as e:
                _add("error", expected_task, jsonl_path, ln_no, "", f"Invalid JSON: {e}")
                continue

            total_samples += 1
            meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
            task = str(meta.get("task_name") or "")
            ev = str(meta.get("evidence_type") or "")
            sample_id = str(entry.get("id") or "")
            if not task:
                _add("error", expected_task, jsonl_path, ln_no, sample_id, "Missing meta.task_name.")
                continue
            if task != expected_task:
                _add("error", task, jsonl_path, ln_no, sample_id, f"Task folder mismatch: folder={expected_task} meta.task_name={task}.")

            allowed = _allowed_evidence_types_for_task(task)
            if not allowed:
                _add("error", task, jsonl_path, ln_no, sample_id, "Unknown task_name (not in task set).")
            elif ev not in allowed:
                _add("error", task, jsonl_path, ln_no, sample_id, f"Wrong evidence_type={ev}; allowed={sorted(allowed)}.")

            conversations = entry.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                _add("error", task, jsonl_path, ln_no, sample_id, "conversations must be a list of length 2.")
                continue
            q = conversations[0].get("value") if isinstance(conversations[0], dict) else None
            a = conversations[1].get("value") if isinstance(conversations[1], dict) else None
            if not isinstance(q, str) or not q.strip():
                _add("error", task, jsonl_path, ln_no, sample_id, "Question must be a non-empty string.")
            if not isinstance(a, str) or not a.strip():
                _add("error", task, jsonl_path, ln_no, sample_id, "Answer must be a non-empty string.")

            if isinstance(q, str) and _has_frame_leak(q):
                _add("error", task, jsonl_path, ln_no, sample_id, "Question contains filename/frame/timestamp leak.")
            if isinstance(a, str) and _has_frame_leak(a):
                _add("error", task, jsonl_path, ln_no, sample_id, "Answer contains filename/frame/timestamp leak.")

            imgs = entry.get("image", [])
            vid = entry.get("video") if isinstance(entry.get("video"), str) and str(entry.get("video")).strip() else None
            if not isinstance(imgs, list):
                _add("error", task, jsonl_path, ln_no, sample_id, "image must be a list.")
                imgs = []


            src = str(meta.get("source_path") or "")
            if src:
                if not os.path.exists(_abs_under_root(src, input_root)):
                    _add("error", task, jsonl_path, ln_no, sample_id, f"source_path not found: {src}")
            else:
                _add("warn", task, jsonl_path, ln_no, sample_id, "meta.source_path is empty.")

            item_dir = meta.get("item_dir")
            if isinstance(item_dir, str) and item_dir.strip() and src:
                item_norm = item_dir.replace("\\", "/").strip().rstrip("/")
                src_norm = str(src).replace("\\", "/").strip()
                if item_norm and src_norm and not src_norm.startswith(item_norm + "/"):
                    _add("warn", task, jsonl_path, ln_no, sample_id, "meta.item_dir is not a prefix of meta.source_path.")

            if bool(require_evidence):
                evidence_files = meta.get("evidence_files") if isinstance(meta.get("evidence_files"), list) else None
                expected_files = list(imgs)
                if vid:
                    expected_files.append(vid)
                if evidence_files is None:
                    _add("error", task, jsonl_path, ln_no, sample_id, "meta.evidence_files must be a list.")
                elif evidence_files != expected_files:
                    _add("error", task, jsonl_path, ln_no, sample_id, "meta.evidence_files must equal image + [video].")

                for p in expected_files:
                    ap = _abs_under_root(str(p), input_root)
                    if not ap or not os.path.exists(ap):
                        _add("error", task, jsonl_path, ln_no, sample_id, f"Evidence file not found: {p}")


                if ev == EVIDENCE_KEYFRAME:
                    if vid:
                        _add("error", task, jsonl_path, ln_no, sample_id, "keyframe_single must not include video.")
                    if len(imgs) != 1:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"keyframe_single must have exactly 1 image (got {len(imgs)}).")
                if ev == EVIDENCE_UNIFORM:
                    if vid:
                        _add("error", task, jsonl_path, ln_no, sample_id, "images_uniform_scene must not include video.")
                    if len(imgs) < 1:
                        _add("error", task, jsonl_path, ln_no, sample_id, "images_uniform_scene must have >=1 image.")
                    if len(imgs) < 2:
                        _add("warn", task, jsonl_path, ln_no, sample_id, "images_uniform_scene has <2 images; consider increasing coverage.")
                if ev in (EVIDENCE_CLIP, EVIDENCE_PREFIX):
                    if not vid:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"{ev} must include video.")
                    if len(imgs) != 0:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"{ev} must not include images.")


            q_str = q if isinstance(q, str) else ""
            a_str = a if isinstance(a, str) else ""


            if task == TASK_08:
                if q_str.strip() != "Looking at the full video, what is the overall high-level goal?":
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_08 question must match the template exactly.")

            if task == TASK_09:
                if "High-level goal:" not in q_str or "From the candidate objects" not in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_09 question must include high-level goal and candidate objects template.")

            if task == TASK_10:
                if ev == EVIDENCE_CLIP and "In this clip, what step goal is being accomplished?" not in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_10(video_clip) question must ask for the step goal of this clip.")
                if ev == EVIDENCE_KEYFRAME and "In this keyframe image, what step goal is being accomplished?" not in q_str:
                    _add("warn", task, jsonl_path, ln_no, sample_id, "Task_10(keyframe_single) question should ask for the step goal of this keyframe image.")

            if task == INTERNAL_PATIENT_IDENTIFICATION_TASK and "In this keyframe image, which object is the primary patient (being acted on)?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Internal patient-identification question must match the expected template.")
            if task == TASK_11 and "In this keyframe image, what is the core action phrase?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_11 question must match the action-phrase template.")
            if task == INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK and "In this keyframe image, what is the interaction hotspot's affordance type?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Internal hotspot-affordance question must match the affordance_type template.")
            if task == INTERNAL_HOTSPOT_MECHANISM_TASK and "Briefly explain the physical mechanism at the interaction hotspot in this keyframe image." not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Internal hotspot-mechanism question must match the mechanism template.")
            if (
                task == TASK_04
                and "First identify the interaction hotspot region, then describe its affordance type and physical mechanism." not in q_str
            ):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_04 question must match the micro-affordance template.")
            if task == TASK_12 and "What action is happening, and what immediate state change is underway?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_12 question must match the state-evolution template.")
            if task == TASK_05 and "In one English sentence, explain the physical causal chain in this keyframe" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_05 question must match the holistic causal-chain template.")
            if task == TASK_13 and "Briefly explain why this step is necessary for achieving the overall goal." not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 question must match the rationale template.")
            if task == TASK_01 and "Before executing this step, what spatial preconditions must hold?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_01 question must match the spatial-precondition template.")
            if task == TASK_02 and "Before executing this step, what affordance/state preconditions must hold?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_02 question must match the affordance-precondition template.")
            if task == TASK_03 and "Is it physically feasible to execute this step right now?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_03 question must match the feasibility template.")
            if (
                task == TASK_14
                and "How does the previous step's outcome support the next step by satisfying its preconditions?" not in q_str
            ):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 question must match the inter-step dependency template.")
            if task == TASK_15 and "What step goal should come next?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 question must match the next-step-from-prefix template.")
            if task == TASK_16 and "infer the missing middle step goals in order." not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_16 question must match the middle-steps infill template.")
            if task == TASK_17 and "Based on this prefix, predict the next K=" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 question must match the next-K-steps template.")
            if task == TASK_18 and (
                "the following bad_plan_steps are proposed as the next steps:" not in q_str
                or "Identify the flaw and repair the plan." not in q_str
                or "Output in the format: FlawStep=...; FlawType=...; Reason=...; Repair:" not in q_str
            ):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 question must include bad_plan_steps and the diagnose+repair output format.")
            if task == TASK_19 and ("What is the most likely outcome if" not in q_str or "do not propose any recovery actions" not in q_str):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_19 question must match the counterfactual-outcome template.")
            if task == TASK_20 and (
                "Failure reason:" not in q_str or "What recovery strategy would most plausibly work?" not in q_str
            ):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_20 question must match the failure-recovery template.")
            if task == INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK and "After applying the recovery strategy, what step goal should come next?" not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Internal next-step-after-recovery question must match the expected template.")

            if task in (TASK_05, TASK_03):
                if "\n" in q_str or "\n" in a_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Single-sentence tasks must not contain newlines.")
                if len(re.findall(r"[.?!]", a_str)) > 1:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Single-sentence task answer seems to contain multiple sentences.")

            if task in (TASK_06, TASK_07):
                if not q_str.strip().startswith('Step goal: "') or "after completing this step" not in q_str.lower():
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_06/07 question must start with Step goal and mention completion.")
                if task == TASK_06 and "After completing this step, what spatial postconditions should hold?" not in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_06 question must match the spatial-postcondition template.")
                if task == TASK_07 and "After completing this step, what affordance/state postconditions should hold?" not in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_07 question must match the affordance-postcondition template.")
                if ev != EVIDENCE_KEYFRAME:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_06/07 must use keyframe_single evidence_type.")

            if task == TASK_09:
                m = re.search(r"From the candidate objects\s+(\[[^\]]*\])", q_str)
                if not m:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_09 question must contain candidate objects JSON list.")
                else:
                    try:
                        cands = json.loads(m.group(1))
                        if not isinstance(cands, list):
                            raise ValueError("candidate is not list")
                        cand_set = {str(x) for x in cands if isinstance(x, str) and str(x).strip()}
                        mentioned = sorted({c for c in cand_set if _answer_mentions_token(a_str, c)})
                        if not mentioned:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_09 answer must mention at least one candidate object token.")


                        ans_snake = sorted(set(_SNAKE_TOKEN_RE.findall(a_str)))
                        bad = [o for o in ans_snake if o not in cand_set]
                        if bad:
                            _add("error", task, jsonl_path, ln_no, sample_id, f"Task_09 answer contains objects not in candidates: {bad[:6]}")
                    except Exception:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_09 candidate objects must be valid JSON list.")

            if task == TASK_18 and not re.match(
                r"^FlawStep=\d+;\s*FlawType=[a-z_]+;\s*Reason=.+?;\s*Repair:\s*1\)\s*\".+\"\s+2\)\s*\".+\"\s+3\)\s*\".+\"\s*$",
                a_str,
            ):
                _add(
                    "error",
                    task,
                    jsonl_path,
                    ln_no,
                    sample_id,
                    "Task_18 answer must follow the format: FlawStep=...; FlawType=...; Reason=...; Repair: 1) \"...\" 2) \"...\" 3) \"...\"",
                )

            if task == TASK_19 and _RECOVERY_SUGGESTION_RE.search(a_str):
                _add("error", task, jsonl_path, ln_no, sample_id, "Counterfactual answers must not propose recovery/action suggestions.")

        f.close()

    return AuditReport(total_samples=total_samples, errors=error_count, warnings=warn_count, issues=issues)


def _audit_output_dir_deep(
    *,
    output_dir: str,
    input_root: str,
    max_issues: int,
    issue_sink: Optional[Any] = None,
) -> AuditReport:

    issues: List[AuditIssue] = []
    error_count = 0
    warn_count = 0
    total_samples = 0

    def _add(sev: str, task: str, file: str, line: int, sid: str, msg: str) -> None:
        nonlocal error_count, warn_count
        if str(sev) == "error":
            error_count += 1
        else:
            warn_count += 1
        iss = AuditIssue(severity=sev, task_name=task, file=file, line=int(line), sample_id=sid, message=msg)
        if issue_sink is not None:
            try:
                issue_sink(iss)
            except Exception:
                pass
        if len(issues) < max(1, int(max_issues)):
            issues.append(iss)

    jsonl_paths = sorted(glob.glob(os.path.join(output_dir, "*", "data.jsonl")))
    if not jsonl_paths:
        _add("error", "", output_dir, 0, "", "No task data.jsonl found under output_dir.")
        return AuditReport(total_samples=0, errors=error_count, warnings=warn_count, issues=issues)

    plan_cache: Dict[str, Dict[str, Any]] = {}

    def _load_plan(src_rel: str) -> Optional[Dict[str, Any]]:
        ap = _abs_under_root(str(src_rel), input_root)
        if not ap:
            return None
        if ap in plan_cache:
            return plan_cache[ap]
        try:
            plan = _read_json(ap)


            _fix_final_schema_identifiers_in_plan(plan)
            plan_cache[ap] = plan
            return plan_cache[ap]
        except Exception:
            return None

    _STEP_GOAL_RE = re.compile(r'Step goal:\s*"([^"]+)"')
    _PREV_GOAL_RE = re.compile(r'Previous step goal:\s*"([^"]+)"')
    _NEXT_GOAL_RE = re.compile(r'Next step goal:\s*"([^"]+)"')
    _LAST_COMPLETED_RE = re.compile(r'Last completed step \(in this prefix\):\s*"([^"]+)"')
    _FAIL_REASON_RE = re.compile(r'Failure reason:\s*"([^"]+)"')
    _RECOVERY_STRAT_RE = re.compile(r'Recovery strategy:\s*"([^"]+)"')

    def _split_inline_numbered(text: str) -> List[str]:
        s = str(text or "").strip()
        matches = list(re.finditer(r"(?:^|\s)(\d+)\)\s*", s))
        if not matches:
            return []
        out_items: List[str] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
            item = s[start:end].strip()
            out_items.append(item)
        return out_items

    def _strip_wrapping_quotes(s: str) -> str:
        t = str(s or "").strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            return t[1:-1].strip()
        return t

    for jsonl_path in jsonl_paths:
        expected_task = os.path.basename(os.path.dirname(jsonl_path))
        try:
            f = open(jsonl_path, "r", encoding="utf-8")
            lines = f
        except Exception as e:
            _add("error", expected_task, jsonl_path, 0, "", f"Failed to read jsonl: {e}")
            continue

        for ln_no, line in enumerate(lines, start=1):
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except Exception as e:
                _add("error", expected_task, jsonl_path, ln_no, "", f"Invalid JSON: {e}")
                continue

            total_samples += 1
            meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
            task = str(meta.get("task_name") or "")
            sample_id = str(entry.get("id") or "")
            if not task:
                _add("error", expected_task, jsonl_path, ln_no, sample_id, "Missing meta.task_name.")
                continue
            if task != expected_task:
                _add("error", task, jsonl_path, ln_no, sample_id, f"Task folder mismatch: folder={expected_task} meta.task_name={task}.")

            conversations = entry.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                _add("error", task, jsonl_path, ln_no, sample_id, "conversations must be a list of length 2.")
                continue
            q = conversations[0].get("value") if isinstance(conversations[0], dict) else None
            a = conversations[1].get("value") if isinstance(conversations[1], dict) else None
            q_str = q if isinstance(q, str) else ""
            a_str = a if isinstance(a, str) else ""
            if not q_str.strip() or not a_str.strip():
                _add("error", task, jsonl_path, ln_no, sample_id, "Question/Answer must be non-empty strings.")
                continue

            src_rel = str(meta.get("source_path") or "").strip()
            item_dir = meta.get("item_dir")
            if isinstance(item_dir, str) and item_dir.strip() and src_rel:
                item_norm = item_dir.replace("\\", "/").strip().rstrip("/")
                src_norm = str(src_rel).replace("\\", "/").strip()
                if item_norm and src_norm and not src_norm.startswith(item_norm + "/"):
                    _add("warn", task, jsonl_path, ln_no, sample_id, "meta.item_dir is not a prefix of meta.source_path.")

            plan = _load_plan(src_rel)
            if not plan:
                _add("error", task, jsonl_path, ln_no, sample_id, f"Failed to load source plan: {src_rel}")
                continue

            steps = _sorted_steps(plan)
            hl = _require_str(plan, "high_level_goal")
            goal_to_steps: Dict[str, List[Dict[str, Any]]] = {}
            for st in steps:
                sg = _require_str(st, "step_goal")
                if sg:
                    goal_to_steps.setdefault(sg, []).append(st)

            def _iter_steps_for_goal(step_goal: str) -> List[Dict[str, Any]]:
                return list(goal_to_steps.get(str(step_goal or "").strip(), []))

            if task == TASK_08:
                hl_norm = _sanitize_space(hl)
                if hl_norm and hl_norm not in _sanitize_space(a_str):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_08 answer must contain the source high_level_goal span.")

            if task == TASK_09:
                exp = list(_extract_key_objects_for_task02(plan) or [])
                exp_set = {str(x).strip() for x in exp if str(x).strip()}
                cand_set: set[str] = set()
                m2 = re.search(r"From the candidate objects\s+(\[[^\]]*\])", q_str)
                if m2:
                    try:
                        cands = json.loads(m2.group(1))
                        if isinstance(cands, list):
                            cand_set = {str(x).strip() for x in cands if isinstance(x, str) and str(x).strip()}
                    except Exception:
                        cand_set = set()
                if cand_set:
                    ans_set = {c for c in cand_set if _answer_mentions_token(a_str, c)}
                else:
                    ans_set = {t for t in exp_set if _answer_mentions_token(a_str, t)}
                if exp_set and ans_set != exp_set:
                    _add(
                        "error",
                        task,
                        jsonl_path,
                        ln_no,
                        sample_id,
                        f"Task_09 answer objects must match extracted key_objects from source JSON (expected={sorted(exp_set)} got={sorted(ans_set)}).",
                    )

            if task == TASK_10:
                all_step_goals = {_require_str(s, "step_goal") for s in steps}
                ans_norm = _sanitize_text_single_line(a_str)
                matches = [x for x in all_step_goals if x and _sanitize_text_single_line(x) in ans_norm]
                if not matches:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_10 answer must contain one of steps[*].step_goal.")
                elif len(matches) > 1:
                    _add("warn", task, jsonl_path, ln_no, sample_id, "Task_10 answer matches multiple step_goal strings; consider keeping it more specific.")

            if task in (
                INTERNAL_PATIENT_IDENTIFICATION_TASK,
                TASK_11,
                INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK,
                INTERNAL_HOTSPOT_MECHANISM_TASK,
                TASK_04,
                TASK_12,
                TASK_05,
                TASK_13,
                TASK_01,
                TASK_02,
                TASK_03,
                TASK_06,
                TASK_07,
                TASK_19,
                TASK_20,
            ):
                m = _STEP_GOAL_RE.search(q_str)
                if not m:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Missing Step goal in question for step-scoped task.")
                else:
                    step_goal = str(m.group(1)).strip()
                    cand_steps = _iter_steps_for_goal(step_goal)
                    if not cand_steps:
                        _add("error", task, jsonl_path, ln_no, sample_id, f'Step goal not found in source JSON: "{step_goal}"')
                        cand_steps = []

                    def _any_match(pred) -> bool:
                        for st in cand_steps:
                            try:
                                if pred(st):
                                    return True
                            except Exception:
                                continue
                        return False

                    if task == INTERNAL_PATIENT_IDENTIFICATION_TASK:
                        def _ok_patient(st: Dict[str, Any]) -> bool:
                            cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
                            pat = _require_str(cc, "patient") if isinstance(cc, dict) else ""
                            return bool(pat) and _answer_mentions_token(a_str, pat)

                        if not _any_match(_ok_patient):
                            _add("error", task, jsonl_path, ln_no, sample_id, "Internal patient-identification answer must contain steps[i].causal_chain.patient for the referenced step.")

                    if task == TASK_11:
                        ans_norm = _sanitize_space(a_str)

                        def _ok_action(st: Dict[str, Any]) -> bool:
                            act = _require_str(st.get("causal_chain") or {}, "action")
                            needle = _sanitize_space(act)
                            return bool(needle) and needle in ans_norm

                        if not _any_match(_ok_action):
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_11 answer must contain steps[i].causal_chain.action for the referenced step.")

                    if task in (INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK, INTERNAL_HOTSPOT_MECHANISM_TASK):
                        key = "affordance_type" if task == INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK else "mechanism"

                        def _ok(st: Dict[str, Any]) -> bool:
                            ans_norm = _sanitize_text_single_line(a_str).lower()
                            relaxed = task in LLM_RELAXED_SPAN_TASKS
                            cfs = st.get("critical_frames") or []
                            for cf in cfs:
                                if not isinstance(cf, dict):
                                    continue
                                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                                v = _require_str(intr, key) if isinstance(intr, dict) else ""
                                if not v:
                                    continue
                                if task == INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK:
                                    if _answer_mentions_token(a_str, v):
                                        return True
                                    continue
                                if _span_present_in_text(v, a_str, relaxed=bool(relaxed)):
                                    return True
                            return False

                        if not _any_match(_ok):
                            _add("error", task, jsonl_path, ln_no, sample_id, f"Internal hotspot-affordance answer must contain the referenced frame interaction.{key} span.")

                    if task == TASK_04:


                        ans_l = a_str.lower()
                        relaxed = task in LLM_RELAXED_SPAN_TASKS

                        def _ok(st: Dict[str, Any]) -> bool:
                            for cf in st.get("critical_frames") or []:
                                if not isinstance(cf, dict):
                                    continue
                                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                                if not isinstance(intr, dict):
                                    continue
                                desc = _require_str(intr, "description").strip()
                                aff_type = _require_str(intr, "affordance_type").strip()
                                mech = _require_str(intr, "mechanism").strip()
                                if not (desc and aff_type and mech):
                                    continue
                                desc_clause = _lowercase_first_alpha(desc.strip().rstrip(".")).lower()
                                mech_clause = _lowercase_first_alpha(_inline_clause(mech)).lower()
                                if aff_type.lower() not in ans_l:
                                    continue
                                if desc_clause and not _span_present_in_text(desc_clause, a_str, relaxed=bool(relaxed)):
                                    continue
                                if mech_clause and not _span_present_in_text(mech_clause, a_str, relaxed=bool(relaxed)):
                                    continue
                                return True
                            return False

                        if not _any_match(_ok):
                            _add(
                                "error",
                                task,
                                jsonl_path,
                                ln_no,
                                sample_id,
                                "Task_04 answer must include hotspot description, affordance_type, and mechanism grounded in the source JSON.",
                            )

                    if task == TASK_12:
                        relaxed = task in LLM_RELAXED_SPAN_TASKS
                        def _ok(st: Dict[str, Any]) -> bool:
                            ans_norm = _sanitize_text_single_line(a_str).lower()
                            for cf in st.get("critical_frames") or []:
                                if not isinstance(cf, dict):
                                    continue
                                asc = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))
                                if _span_present_in_text(asc, a_str, relaxed=bool(relaxed)):
                                    return True
                            return False
                        if not _any_match(_ok):
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_12 answer must contain a critical_frames[*].action_state_change_description span (normalized).")

                    if task == TASK_05:
                        def _expected_task10_anchors(st: Dict[str, Any]) -> List[Tuple[str, str]]:
                            out_anc: List[Tuple[str, str]] = []
                            step_cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
                            agent = _require_str(step_cc, "agent") if isinstance(step_cc, dict) else ""
                            patient = _require_str(step_cc, "patient") if isinstance(step_cc, dict) else ""
                            action = _require_str(step_cc, "action") if isinstance(step_cc, dict) else ""
                            for cf in st.get("critical_frames") or []:
                                if not isinstance(cf, dict):
                                    continue
                                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                                sp_pre_full = fcc.get("causal_precondition_on_spatial") if isinstance(fcc, dict) else ""
                                af_pre_full = fcc.get("causal_precondition_on_affordance") if isinstance(fcc, dict) else ""
                                sp_eff_full = fcc.get("causal_effect_on_spatial") if isinstance(fcc, dict) else ""
                                af_eff_full = fcc.get("causal_effect_on_affordance") if isinstance(fcc, dict) else ""

                                sp_pre_pts = _format_spatial_points(sp_pre_full)
                                af_pre_pts = _format_affordance_points(af_pre_full)
                                sp_eff_pts = _format_spatial_points(sp_eff_full)
                                af_eff_pts = _format_affordance_points(af_eff_full)

                                desc_raw = _require_str(intr, "description") if isinstance(intr, dict) else ""
                                aff_type_raw = _require_str(intr, "affordance_type") if isinstance(intr, dict) else ""
                                mech_raw = _require_str(intr, "mechanism") if isinstance(intr, dict) else ""

                                desc = _inline_clause(desc_raw)
                                aff_type = _inline_clause(aff_type_raw)
                                mech = _inline_clause(mech_raw)

                                asc_raw = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))
                                asc_best = _inline_clause(
                                    _select_best_asc_clause(asc_raw, context=" ".join([desc_raw, aff_type_raw, mech_raw]))
                                )

                                ctx_terms = _normalize_terms(" ".join([step_goal, desc_raw, aff_type_raw, mech_raw, asc_raw]))
                                avoid_terms = _normalize_terms(asc_best) if asc_best else set()
                                avoid_lower = asc_best.lower() if asc_best else ""

                                def _pick_best_point(points: List[str]) -> str:
                                    best = ""
                                    best_score = -10**9
                                    for p in points or []:
                                        clause = _inline_clause(str(p))
                                        if not clause:
                                            continue
                                        clause_terms = _normalize_terms(clause)
                                        score = len(clause_terms & ctx_terms)
                                        if _needs_not_directly_observable(clause):
                                            score -= 1
                                        if avoid_terms:
                                            score -= 0.5 * len(clause_terms & avoid_terms)
                                        if avoid_lower and (clause.lower() in avoid_lower or avoid_lower in clause.lower()):
                                            score -= 2
                                        score -= 0.001 * len(clause)
                                        if score > best_score:
                                            best_score = score
                                            best = clause
                                    return best.strip()

                                sp_pre = _pick_best_point(sp_pre_pts)
                                af_pre = _pick_best_point(af_pre_pts)
                                sp_eff = _pick_best_point(sp_eff_pts)
                                af_eff = _pick_best_point(af_eff_pts)

                                pre_parts: List[str] = []
                                for p in (sp_pre, af_pre):
                                    p = p.strip()
                                    if not p:
                                        continue
                                    p2 = _lowercase_first_alpha(_annotate_observability(p))
                                    if p2 and p2 not in pre_parts:
                                        pre_parts.append(p2)

                                eff_parts: List[str] = []
                                for p in (sp_eff, af_eff):
                                    p = p.strip()
                                    if not p:
                                        continue
                                    p2 = _lowercase_first_alpha(_annotate_observability(p))
                                    if p2 and p2 not in eff_parts:
                                        eff_parts.append(p2)

                                pre_clause = " and ".join(pre_parts).strip()
                                eff_clause = " and ".join(eff_parts).strip()
                                if not (mech and pre_clause and eff_clause):
                                    continue
                                out_anc.append((pre_clause, eff_clause))
                            return out_anc

                        anchors: List[Tuple[str, str]] = []
                        for st in cand_steps:
                            anchors.extend(_expected_task10_anchors(st))
                        if anchors:
                            ans_l = _sanitize_text_single_line(a_str).lower()
                            relaxed = task in LLM_RELAXED_SPAN_TASKS
                            ok = False
                            for pre, eff in anchors:
                                if not (pre and eff):
                                    continue
                                if bool(relaxed):
                                    if _span_present_in_text(pre, a_str, relaxed=True) and _span_present_in_text(eff, a_str, relaxed=True):
                                        ok = True
                                        break
                                    continue
                                if pre.lower() in ans_l and eff.lower() in ans_l:
                                    ok = True
                                    break

                            if len(re.findall(r"[.?!]", _sanitize_text_single_line(a_str))) > 1:
                                ok = False
                            if not ok:
                                _add(
                                    "error",
                                    task,
                                    jsonl_path,
                                    ln_no,
                                    sample_id,
                                    "Task_05 answer must remain grounded in the expected precondition/effect clauses (and be one sentence).",
                                )

                    if task in (TASK_13, TASK_19, TASK_20):
                        key = "rationale" if task == TASK_13 else ("expected_challenge_outcome" if task == TASK_19 else "failure_reflecting")

                        if task == TASK_13:
                            relaxed = task in LLM_RELAXED_SPAN_TASKS

                            def _ok(st: Dict[str, Any]) -> bool:
                                rat = _require_str(st, "rationale")
                                return bool(rat) and _span_present_in_text(rat, a_str, relaxed=bool(relaxed))

                            if not _any_match(_ok):
                                _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 answer must contain steps[i].rationale span.")

                        if task == TASK_19:
                            relaxed = task in LLM_RELAXED_SPAN_TASKS

                            def _ok(st: Dict[str, Any]) -> bool:
                                exp = _clean_counterfactual_outcome(_require_str(st, "expected_challenge_outcome"))
                                return bool(exp) and _span_present_in_text(exp, a_str, relaxed=bool(relaxed))

                            if not _any_match(_ok):
                                _add("error", task, jsonl_path, ln_no, sample_id, "Task_19 answer must contain cleaned steps[i].expected_challenge_outcome span.")

                        if task == TASK_20:
                            relaxed = task in LLM_RELAXED_SPAN_TASKS
                            def _ok(st: Dict[str, Any]) -> bool:
                                fr = st.get("failure_reflecting") if isinstance(st.get("failure_reflecting"), dict) else {}
                                strat = _require_str(fr, "recovery_strategy") if isinstance(fr, dict) else ""
                                s = strat.strip().rstrip()
                                if not s:
                                    return False

                                needle = s.rstrip(".!?").strip()
                                if bool(relaxed):
                                    return bool(needle) and _span_present_in_text(needle, a_str, relaxed=True)
                                needle_l = needle.lower()
                                return needle_l and needle_l in _sanitize_text_single_line(a_str).lower()
                            if not _any_match(_ok):
                                _add(
                                    "error",
                                    task,
                                    jsonl_path,
                                    ln_no,
                                    sample_id,
                                    "Task_20 answer must contain steps[i].failure_reflecting.recovery_strategy (verbatim span).",
                                )

                    if task in (TASK_01, TASK_02, TASK_03, TASK_06, TASK_07):
                        prefer_j = 0 if task in (TASK_01, TASK_02, TASK_03) else 1
                        field_key = (
                            "causal_precondition_on_spatial"
                            if task == TASK_01
                            else (
                                "causal_precondition_on_affordance"
                                if task == TASK_02
                                else (
                                    "causal_effect_on_spatial"
                                    if task == TASK_06
                                    else "causal_effect_on_affordance"
                                )
                            )
                        )

                        def _expected_str(st: Dict[str, Any]) -> str:
                            cfs = st.get("critical_frames") or []
                            if not isinstance(cfs, list) or prefer_j >= len(cfs) or not isinstance(cfs[prefer_j], dict):
                                return ""
                            cf = cfs[prefer_j]
                            fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                            if not isinstance(fcc, dict):
                                return ""
                            if task in (TASK_01, TASK_06):
                                pts = _format_spatial_points(fcc.get(field_key))
                            else:
                                pts = _format_affordance_points(fcc.get(field_key))
                            if task in (TASK_06, TASK_07):
                                pts2 = [_annotate_observability(p) for p in pts if p]
                            else:
                                pts2 = [p for p in pts if p]
                            return _sanitize_space(" ".join(pts2))

                        if task in (TASK_01, TASK_02, TASK_06, TASK_07):
                            relaxed = task in LLM_RELAXED_SPAN_TASKS

                            def _ok(st: Dict[str, Any]) -> bool:
                                exp = _expected_str(st)
                                return bool(exp) and _span_present_in_text(exp, a_str, relaxed=bool(relaxed))

                            if not _any_match(_ok):
                                _add("error", task, jsonl_path, ln_no, sample_id, f"{task} answer must contain the expected causal_chain field span from critical_frames[{prefer_j}].")

                        if task == TASK_03:
                            def _ok(st: Dict[str, Any]) -> bool:
                                cfs = st.get("critical_frames") or []
                                if not isinstance(cfs, list) or len(cfs) < 1 or not isinstance(cfs[0], dict):
                                    return False
                                fcc0 = cfs[0].get("causal_chain") if isinstance(cfs[0].get("causal_chain"), dict) else {}
                                sp_pre_1 = _format_spatial(fcc0.get("causal_precondition_on_spatial"), max_items=1) if isinstance(fcc0, dict) else ""
                                af_pre_1 = _format_affordance(fcc0.get("causal_precondition_on_affordance"), max_items=1) if isinstance(fcc0, dict) else ""
                                if not (sp_pre_1 and af_pre_1):
                                    return False
                                sp_clause = _lowercase_first_alpha(_inline_clause(sp_pre_1))
                                af_clause = _lowercase_first_alpha(_inline_clause(af_pre_1))
                                sp_status = "satisfied"
                                if _needs_not_directly_observable(sp_clause):
                                    sp_status = "not directly observable"
                                af_status = "not directly observable" if _needs_not_directly_observable(af_clause) else "satisfied"
                                prefix = "It is feasible now" if af_status == "satisfied" else "It is likely feasible now"
                                exp = (
                                    f"{prefix} because {sp_clause} (spatial precondition {sp_status}) and "
                                    f"{af_clause} (affordance precondition {af_status})."
                                )
                                exp = _enforce_single_sentence(exp)
                                return _sanitize_space(exp).lower() in _sanitize_space(a_str).lower()

                            if not _any_match(_ok):
                                _add("error", task, jsonl_path, ln_no, sample_id, "Task_03 answer must be derived from critical_frames[0] preconditions.")

            if task == TASK_14:
                m0 = _PREV_GOAL_RE.search(q_str)
                m1 = _NEXT_GOAL_RE.search(q_str)
                if not m0 or not m1:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 question must include Previous/Next step goal.")
                else:
                    sg0 = str(m0.group(1)).strip()
                    sg1 = str(m1.group(1)).strip()
                    idx0 = None
                    for i, st in enumerate(steps):
                        if _require_str(st, "step_goal") == sg0:
                            idx0 = i
                            break
                    if idx0 is None or idx0 + 1 >= len(steps) or _require_str(steps[idx0 + 1], "step_goal") != sg1:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 step goals must correspond to consecutive steps in source JSON.")
                    else:
                        s0 = steps[idx0]
                        s1 = steps[idx0 + 1]
                        cc0 = s0.get("causal_chain") if isinstance(s0.get("causal_chain"), dict) else {}
                        cc1 = s1.get("causal_chain") if isinstance(s1.get("causal_chain"), dict) else {}
                        eff_sp = cc0.get("causal_effect_on_spatial") if isinstance(cc0, dict) else ""
                        eff_af = cc0.get("causal_effect_on_affordance") if isinstance(cc0, dict) else ""
                        pre_sp = cc1.get("causal_precondition_on_spatial") if isinstance(cc1, dict) else ""
                        pre_af = cc1.get("causal_precondition_on_affordance") if isinstance(cc1, dict) else ""
                        eff_cands = [p for p in (_split_numbered_block(eff_sp) + _split_numbered_block(eff_af)) if p]
                        pre_cands = [p for p in (_split_numbered_block(pre_sp) + _split_numbered_block(pre_af)) if p]
                        if eff_cands and pre_cands:
                            best_eff = eff_cands[0]
                            best_pre = pre_cands[0]
                            best_score = -1
                            for e in eff_cands:
                                te = _normalize_terms(e)
                                if not te:
                                    continue
                                for p in pre_cands:
                                    sc = len(te & _normalize_terms(p))
                                    if sc > best_score:
                                        best_score = sc
                                        best_eff = e
                                        best_pre = p
                            if best_score <= 0:
                                best_eff = eff_cands[0]
                                best_pre = pre_cands[0]
                            eff_clause = _lowercase_first_alpha(_inline_clause(_annotate_observability(best_eff)))
                            pre_clause = _lowercase_first_alpha(_inline_clause(_annotate_observability(best_pre)))
                            ans_l = _sanitize_text_single_line(a_str).lower()
                            relaxed = task in LLM_RELAXED_SPAN_TASKS
                            ok = True
                            if not bool(relaxed):
                                if sg0 not in a_str or sg1 not in a_str:
                                    ok = False
                            if eff_clause:
                                if bool(relaxed):
                                    if not _span_present_in_text(eff_clause, a_str, relaxed=True):
                                        ok = False
                                elif eff_clause.lower() not in ans_l:
                                    ok = False
                            if pre_clause:
                                if bool(relaxed):
                                    if not _span_present_in_text(pre_clause, a_str, relaxed=True):
                                        ok = False
                                elif pre_clause.lower() not in ans_l:
                                    ok = False
                            if not ok:
                                _add(
                                    "error",
                                    task,
                                    jsonl_path,
                                    ln_no,
                                    sample_id,
                                    "Task_14 answer must remain grounded in the expected effect/precondition clauses for the consecutive steps.",
                                )
                        else:
                            _add("warn", task, jsonl_path, ln_no, sample_id, "Task_14 deep check skipped (missing causal_chain fields).")

            if task == TASK_15:
                m = _LAST_COMPLETED_RE.search(q_str)
                if not m:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 question must include last completed step goal.")
                else:
                    last_goal = str(m.group(1)).strip()
                    idx0 = None
                    for i, st in enumerate(steps):
                        if _require_str(st, "step_goal") == last_goal:
                            idx0 = i
                            break
                    if idx0 is None or idx0 + 1 >= len(steps):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 last completed step goal not found in source JSON.")
                    else:
                        exp = _require_str(steps[idx0 + 1], "step_goal")
                        if _sanitize_space(exp).lower() not in _sanitize_space(a_str).lower():
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 answer must contain the next step_goal in source JSON.")

            if task == TASK_16:
                if len(steps) >= 3:
                    middle = [_require_str(s, "step_goal") for s in steps[1:-1] if _require_str(s, "step_goal")]
                    ans_norm = _sanitize_text_single_line(a_str).lower()
                    ok = True
                    pos = -1
                    for sg in middle:
                        needle = _sanitize_text_single_line(sg).lower()
                        if not needle:
                            continue
                        idx = ans_norm.find(needle)
                        if idx < 0 or idx < pos:
                            ok = False
                            break
                        pos = idx
                    if not ok:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_16 answer must contain all middle step goals in order.")

            if task == TASK_17:
                m = _LAST_COMPLETED_RE.search(q_str)
                km = re.search(r"next K=(\d+)", q_str)
                if not m or not km:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 question must include last completed step goal and K.")
                else:
                    last_goal = str(m.group(1)).strip()
                    try:
                        k = int(km.group(1))
                    except Exception:
                        k = 0
                    idx0 = None
                    for i, st in enumerate(steps):
                        if _require_str(st, "step_goal") == last_goal:
                            idx0 = i
                            break
                    if idx0 is None or k <= 0:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 last completed goal not found or invalid K.")
                    else:
                        gold = [_require_str(s, "step_goal") for s in steps[idx0 + 1 : idx0 + 1 + k] if _require_str(s, "step_goal")]
                        ans_norm = _sanitize_text_single_line(a_str).lower()
                        ok = True
                        pos = -1
                        for sg in gold:
                            needle = _sanitize_text_single_line(sg).lower()
                            if not needle:
                                continue
                            idx = ans_norm.find(needle)
                            if idx < 0 or idx < pos:
                                ok = False
                                break
                            pos = idx
                        if not ok:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 answer must contain the next K step goals in order.")

            if task == TASK_18:
                seg = q_str
                m_seg = re.search(
                    r"the following bad_plan_steps are proposed as the next steps:\s*(.*?)\s*Identify the flaw and repair the plan\.",
                    q_str,
                )
                if m_seg:
                    seg = str(m_seg.group(1) or "").strip()
                q_items = re.findall(r'\d+\)\s*"([^"]+)"', seg)
                if len(q_items) != 3:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 question must include exactly 3 bad_plan_steps.")
                    q_items = []
                step_goals = [_require_str(s, "step_goal") for s in steps]

                fm = re.search(r"FlawStep\s*=\s*(\d+)", a_str)
                tm = re.search(r"FlawType\s*=\s*([a-z_]+)", a_str)
                if not fm or not tm:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 answer must include FlawStep=... and FlawType=....")
                    flaw_step = 0
                else:
                    try:
                        flaw_step = int(fm.group(1))
                    except Exception:
                        flaw_step = 0
                flaw_type = str(tm.group(1)).strip() if tm else ""
                if flaw_type and flaw_type not in {"goal_inconsistent", "precondition_missing", "redundant_step"}:
                    _add("error", task, jsonl_path, ln_no, sample_id, f"Task_18 FlawType must be one of goal_inconsistent/precondition_missing/redundant_step (got {flaw_type}).")
                if flaw_step not in (1, 2, 3):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 FlawStep must be 1, 2, or 3.")

                ans_steps = re.findall(r'\d+\)\s*"([^"]+)"', a_str)
                if len(ans_steps) != 3:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 answer must include exactly 3 repaired steps under Repair:.")
                    ans_steps = []
                if ans_steps and any(it not in step_goals for it in ans_steps):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 repaired steps must come from steps[*].step_goal.")



                if q_items and ans_steps:
                    best_slice: Optional[List[str]] = None
                    best_mismatch_idx: Optional[int] = None
                    for i in range(len(step_goals) - 2):
                        sl = step_goals[i : i + 3]
                        mism = [j for j in range(3) if q_items[j] != sl[j]]
                        if len(mism) == 1:
                            best_slice = sl
                            best_mismatch_idx = mism[0]
                            break
                    if best_slice is None or best_mismatch_idx is None:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 bad_plan_steps must differ from a contiguous 3-step slice at exactly one position.")
                    else:
                        if ans_steps != best_slice:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 Repair steps must equal the gold contiguous 3-step slice from the source plan.")
                        if flaw_step and flaw_step != best_mismatch_idx + 1:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 FlawStep must point to the mismatched bad_plan_steps position.")

            if task == INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK:
                rm = _FAIL_REASON_RE.search(q_str)
                sm = _RECOVERY_STRAT_RE.search(q_str)
                if not rm or not sm:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Internal next-step-after-recovery question must include failure reason and recovery strategy.")
                else:
                    reason = str(rm.group(1)).strip()
                    strat = str(sm.group(1)).strip()
                    matched = False
                    for i, st in enumerate(steps):
                        fr = st.get("failure_reflecting") if isinstance(st.get("failure_reflecting"), dict) else {}
                        if not isinstance(fr, dict):
                            continue
                        if _require_str(fr, "reason") == reason and _require_str(fr, "recovery_strategy") == strat:
                            exp = _require_str(st, "step_goal")
                            if _sanitize_space(exp).lower() not in _sanitize_space(a_str).lower():
                                _add("error", task, jsonl_path, ln_no, sample_id, "Internal next-step-after-recovery answer must contain the referenced step_goal (retry current step).")
                            matched = True
                            break
                    if not matched:
                        _add("error", task, jsonl_path, ln_no, sample_id, "Internal next-step-after-recovery failure reason/strategy pair not found in source JSON.")

        f.close()

    return AuditReport(total_samples=total_samples, errors=error_count, warnings=warn_count, issues=issues)


def _sorted_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    def _to_int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0
    return sorted(steps, key=lambda s: _to_int(s.get("step_id")))


def _require_str(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    return v.strip() if isinstance(v, str) else ""


def _frame_index(cf: Dict[str, Any]) -> Optional[int]:
    try:
        v = cf.get("frame_index")
        return int(v)
    except Exception:
        return None


def _make_task01(item_dir: str, plan: Dict[str, Any], input_root: str, *, uniform_k: int, attach_evidence: bool) -> Optional[Sample]:
    steps = _sorted_steps(plan)
    if not steps:
        return None
    hl = _require_str(plan, "high_level_goal")
    if not hl:
        return None

    q = "Looking at the full video, what is the overall high-level goal?"
    a = hl

    if not bool(attach_evidence):
        evidence_type = EVIDENCE_PREFIX
        images = []
        video_rel = None
    else:
        last_step_id = int(steps[-1].get("step_id", 0) or 0)
        video = _resolve_video_prefix(item_dir, last_step_id)
        if video:
            evidence_type = EVIDENCE_PREFIX
            images = []
            video_rel = _safe_relpath(video, input_root)
        else:
            sampled = _list_sampled_frames(item_dir)
            if not sampled:
                return None
            evidence_type = EVIDENCE_UNIFORM
            imgs = _pick_uniform(sampled, uniform_k)
            if not imgs:
                return None
            images = [_safe_relpath(p, input_root) for p in imgs]
            video_rel = None
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    return Sample(
        task_name=TASK_08,
        evidence_type=evidence_type,
        image=images,
        video=video_rel,
        question=q,
        answer=a,
        source_path=source_rel,
        llm_fields={"high_level_goal": hl},
    )


def _make_task02(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    uniform_k: int,
    rng: random.Random,
    *,
    attach_evidence: bool,
) -> Optional[Sample]:
    hl = _require_str(plan, "high_level_goal")
    if not hl:
        return None
    label_objs = _extract_key_objects_for_task02(plan)
    if not label_objs:
        return None

    candidates = list(label_objs)
    cand_lower = {str(x).strip().lower() for x in candidates if isinstance(x, str) and x.strip()}
    target_len = max(10, min(14, len(label_objs) + 6))
    hl_lower = hl.lower()
    label_lower = [str(x).strip().lower() for x in label_objs if isinstance(x, str) and x.strip()]
    distractors = list(_DISTRACTOR_OBJECTS)
    rng.shuffle(distractors)
    for d in distractors:
        dl = str(d or "").strip().lower()
        if not dl or dl in cand_lower:
            continue


        if any(dl in o for o in label_lower):
            continue
        if "_" in dl:
            parts = [re.escape(p) for p in dl.split("_") if p]
            if parts and re.search(r"\b" + r"\s+".join(parts) + r"\b", hl_lower):
                continue
        else:
            if re.search(rf"\b{re.escape(dl)}\b", hl_lower):
                continue
        candidates.append(str(d))
        cand_lower.add(dl)
        if len(candidates) >= target_len:
            break
    rng.shuffle(candidates)

    q = (
        f'High-level goal: "{hl}" From the candidate objects {json.dumps(candidates)}, '
        "name the key objects that are directly relevant to achieving the goal."
    )
    def _join_quoted(items: Sequence[str]) -> str:
        xs = [f'"{str(x).strip()}"' for x in items if str(x).strip()]
        if not xs:
            return ""
        if len(xs) == 1:
            return xs[0]
        if len(xs) == 2:
            return f"{xs[0]} and {xs[1]}"
        return ", ".join(xs[:-1]) + f", and {xs[-1]}"

    joined = _join_quoted(label_objs)
    a = _sanitize_space(f"The key objects directly relevant to the goal are {joined}.") if joined else ""
    if not a:
        return None

    images: List[str] = []
    if bool(attach_evidence):
        sampled = _list_sampled_frames(item_dir)
        if not sampled:
            return None
        imgs = _pick_uniform(sampled, uniform_k)
        if not imgs:
            return None
        images = [_safe_relpath(p, input_root) for p in imgs]
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    return Sample(
        task_name=TASK_09,
        evidence_type=EVIDENCE_UNIFORM,
        image=images,
        video=None,
        question=q,
        answer=a,
        source_path=source_rel,
        llm_fields={"high_level_goal": hl, "key_objects": list(label_objs)},
    )


def _make_task03(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    require_video: bool,
    attach_evidence: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if not hl or len(steps) < 1:
        return []

    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    for st in steps:
        step_id = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if step_id <= 0 or not step_goal:
            continue
        if not bool(attach_evidence):
            evidence_type = EVIDENCE_CLIP
            q = f'Context: High-level goal: "{hl}" In this clip, what step goal is being accomplished?'
            video_rel = None
            images = []
        else:
            clip = _resolve_video_clip(item_dir, step_id)
            if require_video and not clip:
                continue

            if clip:
                evidence_type = EVIDENCE_CLIP
                q = f'Context: High-level goal: "{hl}" In this clip, what step goal is being accomplished?'
                video_rel = _safe_relpath(clip, input_root)
                images = []
            else:
                cfs = st.get("critical_frames") or []
                thumb = None
                if isinstance(cfs, list) and cfs:
                    last_cf = cfs[-1] if isinstance(cfs[-1], dict) else None
                    fi = _frame_index(last_cf) if last_cf else None
                    if fi is not None:
                        thumb = _find_keyframe_image(item_dir, step_id, fi)
                if not thumb:
                    continue
                evidence_type = EVIDENCE_KEYFRAME
                q = f'Context: High-level goal: "{hl}" In this keyframe image, what step goal is being accomplished?'
                video_rel = None
                images = [_safe_relpath(thumb, input_root)]
        a = step_goal
        out.append(
            Sample(
                task_name=TASK_10,
                evidence_type=evidence_type,
                image=images,
                video=video_rel,
                question=q,
                answer=a,
                source_path=source_rel,
                llm_fields={"high_level_goal": hl, "step_id": step_id, "step_goal": step_goal},
            )
        )
    return out


def _keyframe_for_task(
    item_dir: str,
    step: Dict[str, Any],
    *,
    prefer_j: int,
    require_image: bool,
) -> Optional[Tuple[int, Dict[str, Any], str]]:
    step_id = int(step.get("step_id", 0) or 0)
    if step_id <= 0:
        return None
    cfs = step.get("critical_frames") or []
    if not isinstance(cfs, list) or len(cfs) < 1:
        return None
    j = int(prefer_j)
    if j < 0 or j >= len(cfs) or not isinstance(cfs[j], dict):
        j = 0
    cf = cfs[j]
    fi = _frame_index(cf)
    if fi is None:
        return None
    if not bool(require_image):
        return step_id, cf, ""
    img = _find_keyframe_image(item_dir, step_id, fi)
    if not img:
        return None
    return step_id, cf, img


_LATENT_KEYWORDS = (
    "not directly observable",
    "internal",
    "inside",
    "friction",
    "texture",
    "static",
    "circuit",
    "electrical",
    "magnetic",
    "chemical",
    "temperature",
    "pressure",
    "material",
    "viscos",
    "sufficient",
    "enough",
)

_RECOVERY_SUGGESTION_RE = re.compile(
    r"(?:,?\s*(?:and|so)\s+)?"
    r"(?:the\s+)?(?:cook|person|agent|operator|user|they|you)\s+"
    r"(?:would\s+)?(?:need|needs|must|should|have\s+to)\b",
    flags=re.IGNORECASE,
)


def _inline_clause(text: str) -> str:
    s = _sanitize_text_single_line(text)
    s = re.sub(r"[.?!]+\s+", "; ", s)
    s = s.strip().strip(";")
    s = re.sub(r"[;:,]+$", "", s).strip()
    s = re.sub(r"[.?!]+$", "", s).strip()
    return s


def _select_best_asc_clause(asc: str, *, context: str) -> str:

    s = _sanitize_text_single_line(asc)
    s = _KEY_MOMENT_PREFIX_RE.sub("", s).strip()
    if not s:
        return ""
    clauses = [c.strip() for c in s.split(";") if c.strip()]
    if not clauses:
        clauses = [s]
    ctx_terms = _normalize_terms(str(context or ""))
    if not ctx_terms:
        return clauses[0].strip()
    best = clauses[0]
    best_score = -1
    for c in clauses:
        sc = len(_normalize_terms(c) & ctx_terms)
        if sc > best_score:
            best_score = sc
            best = c
    return best.strip()


def _needs_not_directly_observable(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return False
    if "not directly observable" in s:
        return True
    return any(k in s for k in _LATENT_KEYWORDS[1:])


def _annotate_observability(text: str, *, force: bool = False, from_clip: bool = False) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    if "not directly observable" in s.lower():
        return s
    if force or _needs_not_directly_observable(s):
        suffix = "not directly observable from this clip" if from_clip else "not directly observable"
        m = re.match(r"^(.*?)([.?!])$", s)
        if m:
            return f"{m.group(1)} ({suffix}){m.group(2)}"
        return f"{s} ({suffix})"
    return s


def _clean_counterfactual_outcome(outcome: str) -> str:

    sent = _enforce_single_sentence(str(outcome or "").strip())
    if not sent:
        return ""
    m = _RECOVERY_SUGGESTION_RE.search(sent)
    if not m:
        return sent

    cut = sent[: m.start()].rstrip(" ,;:").strip()
    if cut:
        if cut[-1] not in ".?!":
            cut = cut + "."
        return cut


    tail = ""
    comma = sent.find(",", max(0, int(m.end())))
    if comma != -1:
        tail = sent[comma + 1 :].strip()
    else:
        tail = sent[m.end() :].strip()
    tail = tail.lstrip(" ,;:").strip()
    tail = re.sub(r"^(?:and\s+)+", "", tail, flags=re.IGNORECASE).strip()
    if not tail:
        return ""


    lower = tail.lower()
    if lower.startswith("risking "):
        tail = "There is a risk of " + tail[len("risking ") :].lstrip()
    elif lower.startswith("delaying "):
        tail = "This could delay " + tail[len("delaying ") :].lstrip()
    elif lower.startswith("leading to "):
        tail = "This could lead to " + tail[len("leading to ") :].lstrip()
    elif lower.startswith("causing "):
        tail = "This could cause " + tail[len("causing ") :].lstrip()

    tail = tail.strip()
    if not tail:
        return ""
    if tail[-1] not in ".?!":
        tail = tail + "."

    if _RECOVERY_SUGGESTION_RE.search(tail):
        return ""
    return tail


def _lowercase_first_alpha(text: str) -> str:
    s = str(text or "")
    if not s:
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def _humanize_snake_token(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return s
    if "_" in s and re.fullmatch(r"[A-Za-z0-9_]+", s):
        return s.replace("_", " ")
    return s


def _with_definite_article(token: str) -> str:
    s = _humanize_snake_token(token)
    if not s:
        return s
    if re.match(r"^(the|a|an)\\b", s, flags=re.IGNORECASE):
        return s
    return f"the {s}"


def _enforce_single_sentence(text: str) -> str:
    s = _sanitize_text_single_line(text)
    if not s:
        return s
    punct = [m.start() for m in re.finditer(r"[.?!]", s)]
    if len(punct) <= 1:
        if s[-1] not in ".?!":
            return s + "."
        return s
    chars = list(s)
    for i in punct[:-1]:
        chars[i] = ";"
    s2 = _sanitize_text_single_line("".join(chars))
    if s2[-1] not in ".?!":
        s2 = s2 + "."
    return s2


def _make_task04_to_16(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    if not steps:
        return []

    out: List[Sample] = []

    def _img_list(path: str) -> List[str]:
        if not bool(attach_evidence):
            return []
        p = str(path or "").strip()
        if not p:
            return []
        return [_safe_relpath(p, input_root)]

    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue

        step_cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        agent = _require_str(step_cc, "agent") if isinstance(step_cc, dict) else ""
        patient = _require_str(step_cc, "patient") if isinstance(step_cc, dict) else ""
        action = _require_str(step_cc, "action") if isinstance(step_cc, dict) else ""

        k0 = _keyframe_for_task(item_dir, st, prefer_j=0, require_image=bool(attach_evidence))
        k1 = _keyframe_for_task(item_dir, st, prefer_j=1, require_image=bool(attach_evidence))


        if k0:
            _, _, img0 = k0
            if patient:
                q = f'Context: Step goal: "{step_goal}" In this keyframe image, which object is the primary patient (being acted on)?'
                pat = patient.strip()
                quoted = f'"{pat}"' if pat and '"' not in pat else pat
                a04 = _sanitize_space(f"The primary patient object being acted on is {quoted}.") if quoted else ""
                if a04:
                    out.append(
                        Sample(
                            INTERNAL_PATIENT_IDENTIFICATION_TASK,
                            EVIDENCE_KEYFRAME,
                            _img_list(img0),
                            None,
                            q,
                            a04,
                            source_rel,
                            llm_fields={"step_goal": step_goal, "patient": pat},
                        )
                    )
            if action:
                q = f'Context: Step goal: "{step_goal}" In this keyframe image, what is the core action phrase?'
                act = action.strip()
                quoted = f'"{act}"' if act and '"' not in act else act
                a05 = _sanitize_space(f"The action phrase in this keyframe is {quoted}.") if quoted else ""
                if a05:
                    out.append(
                        Sample(
                            TASK_11,
                            EVIDENCE_KEYFRAME,
                            _img_list(img0),
                            None,
                            q,
                            a05,
                            source_rel,
                            llm_fields={"step_goal": step_goal, "action": act},
                        )
                    )


        frames_for_hotspot: List[Tuple[Dict[str, Any], str]] = []
        if k0:
            _, cf0, img0 = k0
            frames_for_hotspot.append((cf0, img0))
        if k1:
            _, cf1, img1 = k1
            if not frames_for_hotspot or frames_for_hotspot[-1][1] != img1:
                frames_for_hotspot.append((cf1, img1))

        for cf, img in frames_for_hotspot:
            intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
            hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
            aff_type = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
            mech = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""
            desc = _require_str(hotspot, "description") if isinstance(hotspot, dict) else ""
            asc = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))

            if aff_type:
                q = f'Context: Step goal: "{step_goal}" In this keyframe image, what is the interaction hotspot\'s affordance type?'
                aff = aff_type.strip()
                quoted = f'"{aff}"' if aff and '"' not in aff else aff
                a06 = _sanitize_space(f"The hotspot affordance type is {quoted}.") if quoted else ""
                if a06:
                    out.append(
                        Sample(
                            INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK,
                            EVIDENCE_KEYFRAME,
                            _img_list(img),
                            None,
                            q,
                            a06,
                            source_rel,
                            llm_fields={"step_goal": step_goal, "affordance_type": aff},
                        )
                    )
            if mech:
                q = f'Context: Step goal: "{step_goal}" Briefly explain the physical mechanism at the interaction hotspot in this keyframe image.'
                mech_s = mech.strip()
                if mech_s:
                    out.append(
                        Sample(
                            INTERNAL_HOTSPOT_MECHANISM_TASK,
                            EVIDENCE_KEYFRAME,
                            _img_list(img),
                            None,
                            q,
                            mech_s,
                            source_rel,
                            llm_fields={"step_goal": step_goal, "mechanism": mech_s},
                        )
                    )
            if desc and aff_type and mech:
                q = f'Context: Step goal: "{step_goal}" First identify the interaction hotspot region, then describe its affordance type and physical mechanism.'
                mech_clause = _lowercase_first_alpha(_inline_clause(mech))
                desc_clause = _lowercase_first_alpha(desc.strip().rstrip("."))


                desc_l = desc_clause.lower()
                if re.match(r"^(the|a|an)\\b", desc_l):
                    desc_np = desc_clause
                else:
                    desc_np = "the " + desc_clause
                if re.match(r"^(left|right)\\b", desc_l) or desc_l.startswith(
                    ("hand ", "hands ", "finger ", "fingers ", "thumb ", "person ", "actor ", "human ")
                ):

                    intro = "" if re.match(r"^(the|a|an)\\b", desc_l) else "the "
                    desc_text = f"the point where {intro}{desc_clause}"
                else:

                    desc_text = desc_np
                a = _sanitize_space(
                    f"The hotspot is {desc_text}."
                    f" It affords {aff_type.strip()}, and the mechanism is that {mech_clause}."
                )
                out.append(
                    Sample(
                        TASK_04,
                        EVIDENCE_KEYFRAME,
                        _img_list(img),
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "step_goal": step_goal,
                            "hotspot_description": desc.strip(),
                            "affordance_type": aff_type.strip(),
                            "mechanism": mech.strip(),
                        },
                    )
                )
            if asc:
                q = f'Context: Step goal: "{step_goal}" What action is happening, and what immediate state change is underway?'
                asc_s = asc.strip()
                if asc_s:
                    out.append(
                        Sample(
                            TASK_12,
                            EVIDENCE_KEYFRAME,
                            _img_list(img),
                            None,
                            q,
                            asc_s,
                            source_rel,
                            llm_fields={"step_goal": step_goal, "action_state_change_description": asc_s},
                        )
                    )


        frames_for_task10: List[Tuple[Dict[str, Any], str]] = []
        if k0:
            _, cf0, img0 = k0
            frames_for_task10.append((cf0, img0))
        if k1:
            _, cf1, img1 = k1
            fi0 = _frame_index(frames_for_task10[0][0]) if frames_for_task10 else None
            fi1 = _frame_index(cf1)
            if not frames_for_task10 or (fi0 is None or fi1 is None or fi0 != fi1):
                frames_for_task10.append((cf1, img1))

        for cf, img in frames_for_task10:

                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}

                sp_pre_full = fcc.get("causal_precondition_on_spatial") if isinstance(fcc, dict) else ""
                af_pre_full = fcc.get("causal_precondition_on_affordance") if isinstance(fcc, dict) else ""
                sp_eff_full = fcc.get("causal_effect_on_spatial") if isinstance(fcc, dict) else ""
                af_eff_full = fcc.get("causal_effect_on_affordance") if isinstance(fcc, dict) else ""

                sp_pre_pts = _format_spatial_points(sp_pre_full)
                af_pre_pts = _format_affordance_points(af_pre_full)
                sp_eff_pts = _format_spatial_points(sp_eff_full)
                af_eff_pts = _format_affordance_points(af_eff_full)

                hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
                desc_raw = _require_str(hotspot, "description") if isinstance(hotspot, dict) else ""
                aff_type_raw = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
                mech_raw = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""

                desc = _inline_clause(desc_raw)
                aff_type = _inline_clause(aff_type_raw)
                mech = _inline_clause(mech_raw)

                asc_raw = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))
                asc_best = _inline_clause(
                    _select_best_asc_clause(asc_raw, context=" ".join([desc_raw, aff_type_raw, mech_raw]))
                )

                ctx_terms = _normalize_terms(" ".join([step_goal, desc_raw, aff_type_raw, mech_raw, asc_raw]))
                avoid_terms = _normalize_terms(asc_best) if asc_best else set()
                avoid_lower = asc_best.lower() if asc_best else ""

                def _pick_best_point(points: List[str]) -> str:
                    best = ""
                    best_score = -10**9
                    for p in points or []:
                        clause = _inline_clause(str(p))
                        if not clause:
                            continue
                        clause_terms = _normalize_terms(clause)
                        score = len(clause_terms & ctx_terms)
                        if _needs_not_directly_observable(clause):
                            score -= 1
                        if avoid_terms:
                            score -= 0.5 * len(clause_terms & avoid_terms)
                        if avoid_lower and (clause.lower() in avoid_lower or avoid_lower in clause.lower()):
                            score -= 2
                        score -= 0.001 * len(clause)
                        if score > best_score:
                            best_score = score
                            best = clause
                    return best.strip()

                sp_pre = _pick_best_point(sp_pre_pts)
                af_pre = _pick_best_point(af_pre_pts)
                sp_eff = _pick_best_point(sp_eff_pts)
                af_eff = _pick_best_point(af_eff_pts)

                pre_parts: List[str] = []
                for p in (sp_pre, af_pre):
                    p = p.strip()
                    if not p:
                        continue
                    p2 = _lowercase_first_alpha(_annotate_observability(p))
                    if p2 and p2 not in pre_parts:
                        pre_parts.append(p2)

                eff_parts: List[str] = []
                for p in (sp_eff, af_eff):
                    p = p.strip()
                    if not p:
                        continue
                    p2 = _lowercase_first_alpha(_annotate_observability(p))
                    if p2 and p2 not in eff_parts:
                        eff_parts.append(p2)

                pre_clause = " and ".join(pre_parts).strip()
                eff_clause = " and ".join(eff_parts).strip()

                if not (mech and pre_clause and eff_clause):
                    continue

                q = (
                    f'Context: Step goal: "{step_goal}" '
                    "In one English sentence, explain the physical causal chain in this keyframe—spatial setup, affordance/mechanism, and immediate local effects."
                )

                hotspot_clause = ""
                if desc:
                    hotspot_clause = f" at the hotspot ({desc})"
                elif aff_type:
                    hotspot_clause = f" at the hotspot (affordance: {aff_type})"

                action_clause = asc_best
                if not action_clause and agent and action and patient:
                    agent_phrase = _with_definite_article(agent)
                    patient_phrase = _with_definite_article(patient)
                    action_clause = _inline_clause(f"{agent_phrase} {action} {patient_phrase}")
                action_clause = _lowercase_first_alpha(action_clause).strip()
                if not action_clause:
                    action_clause = "the interaction occurs"

                afford_clause = f", which affords {aff_type}" if aff_type else ""
                mech2 = _lowercase_first_alpha(mech)
                a = f"When {pre_clause}, {action_clause}{hotspot_clause}{afford_clause}; because {mech2}, {eff_clause}."
                a = _enforce_single_sentence(a)

                out.append(
                    Sample(
                        TASK_05,
                        EVIDENCE_KEYFRAME,
                        _img_list(img),
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "high_level_goal": hl,
                            "step_goal": step_goal,
                            "agent": agent,
                            "action": action,
                            "patient": patient,
                            "action_state_change_description": asc_raw,
                            "spatial_preconditions": sp_pre_full,
                            "affordance_preconditions": af_pre_full,
                            "hotspot_description": desc_raw,
                            "affordance_type": aff_type_raw,
                            "mechanism": mech_raw,
                            "spatial_effects": sp_eff_full,
                            "affordance_effects": af_eff_full,
                        },
                    )
                )


        rationale = _require_str(st, "rationale")
        if k0 and hl and rationale:
            _, _, img0 = k0
            q = f'High-level goal: "{hl}" Step goal: "{step_goal}" Briefly explain why this step is necessary for achieving the overall goal.'
            out.append(
                Sample(
                    TASK_13,
                    EVIDENCE_KEYFRAME,
                    _img_list(img0),
                    None,
                    q,
                    rationale,
                    source_rel,
                    llm_fields={"high_level_goal": hl, "step_goal": step_goal, "rationale": rationale},
                )
            )


        if k0:
            _, cf0, img0 = k0
            fcc0 = cf0.get("causal_chain") if isinstance(cf0.get("causal_chain"), dict) else {}
            sp_pre_pts = _format_spatial_points(fcc0.get("causal_precondition_on_spatial")) if isinstance(fcc0, dict) else []
            af_pre_pts = _format_affordance_points(fcc0.get("causal_precondition_on_affordance")) if isinstance(fcc0, dict) else []
            sp_pre_all = _sanitize_space(" ".join([p for p in sp_pre_pts if p]))
            af_pre_all = _sanitize_space(" ".join([p for p in af_pre_pts if p]))
            if sp_pre_all:
                q = f'Step goal: "{step_goal}" Before executing this step, what spatial preconditions must hold?'
                out.append(
                    Sample(
                        TASK_01,
                        EVIDENCE_KEYFRAME,
                        _img_list(img0),
                        None,
                        q,
                        sp_pre_all,
                        source_rel,
                        llm_fields={"step_goal": step_goal, "spatial_preconditions": sp_pre_all},
                    )
                )
            if af_pre_all:
                q = f'Step goal: "{step_goal}" Before executing this step, what affordance/state preconditions must hold?'
                out.append(
                    Sample(
                        TASK_02,
                        EVIDENCE_KEYFRAME,
                        _img_list(img0),
                        None,
                        q,
                        af_pre_all,
                        source_rel,
                        llm_fields={"step_goal": step_goal, "affordance_preconditions": af_pre_all},
                    )
                )

            sp_pre_1 = _format_spatial(fcc0.get("causal_precondition_on_spatial"), max_items=1) if isinstance(fcc0, dict) else ""
            af_pre_1 = _format_affordance(fcc0.get("causal_precondition_on_affordance"), max_items=1) if isinstance(fcc0, dict) else ""
            if sp_pre_1 and af_pre_1:
                sp_clause = _lowercase_first_alpha(_inline_clause(sp_pre_1))
                af_clause = _lowercase_first_alpha(_inline_clause(af_pre_1))
                sp_status = "satisfied"
                if _needs_not_directly_observable(sp_clause):
                    sp_status = "not directly observable"
                af_status = "not directly observable" if _needs_not_directly_observable(af_clause) else "satisfied"
                prefix = "It is feasible now" if af_status == "satisfied" else "It is likely feasible now"
                a = (
                    f"{prefix} because {sp_clause} (spatial precondition {sp_status}) and "
                    f"{af_clause} (affordance precondition {af_status})."
                )
                a = _enforce_single_sentence(a)
                q = (
                    f'Step goal: "{step_goal}" Is it physically feasible to execute this step right now? '
                    "Answer in one English sentence, and justify using one spatial precondition and one affordance precondition, "
                    "each labeled as satisfied/violated/not directly observable in this frame."
                )
                out.append(
                    Sample(
                        TASK_03,
                        EVIDENCE_KEYFRAME,
                        _img_list(img0),
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "step_goal": step_goal,
                            "spatial_precondition": sp_pre_1,
                            "affordance_precondition": af_pre_1,
                            "draft_answer": a,
                        },
                    )
                )


        if k1:
            _, cf1, img1 = k1
            fcc1 = cf1.get("causal_chain") if isinstance(cf1.get("causal_chain"), dict) else {}
            sp_eff_pts = _format_spatial_points(fcc1.get("causal_effect_on_spatial")) if isinstance(fcc1, dict) else []
            af_eff_pts = _format_affordance_points(fcc1.get("causal_effect_on_affordance")) if isinstance(fcc1, dict) else []

            if sp_eff_pts:
                q = f'Step goal: "{step_goal}" After completing this step, what spatial postconditions should hold?'
                a = _sanitize_space(" ".join([_annotate_observability(p) for p in sp_eff_pts]))
                out.append(
                    Sample(
                        TASK_06,
                        EVIDENCE_KEYFRAME,
                        _img_list(img1),
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={"step_goal": step_goal, "spatial_postconditions": a},
                    )
                )
            if af_eff_pts:
                q = f'Step goal: "{step_goal}" After completing this step, what affordance/state postconditions should hold?'
                a = _sanitize_space(" ".join([_annotate_observability(p) for p in af_eff_pts]))
                out.append(
                    Sample(
                        TASK_07,
                        EVIDENCE_KEYFRAME,
                        _img_list(img1),
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={"step_goal": step_goal, "affordance_postconditions": a},
                    )
                )

    return out


def _make_task17(item_dir: str, plan: Dict[str, Any], input_root: str, *, attach_evidence: bool) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2 or not hl:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)

    def _img_list(path: str) -> List[str]:
        if not bool(attach_evidence):
            return []
        p = str(path or "").strip()
        if not p:
            return []
        return [_safe_relpath(p, input_root)]

    for i in range(len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]
        sid0 = int(s0.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue
        cc0 = s0.get("causal_chain") if isinstance(s0.get("causal_chain"), dict) else {}
        cc1 = s1.get("causal_chain") if isinstance(s1.get("causal_chain"), dict) else {}
        eff_sp = cc0.get("causal_effect_on_spatial") if isinstance(cc0, dict) else ""
        eff_af = cc0.get("causal_effect_on_affordance") if isinstance(cc0, dict) else ""
        pre_sp = cc1.get("causal_precondition_on_spatial") if isinstance(cc1, dict) else ""
        pre_af = cc1.get("causal_precondition_on_affordance") if isinstance(cc1, dict) else ""

        if not (_split_numbered_block(eff_sp) or _split_numbered_block(eff_af)):
            continue
        if not (_split_numbered_block(pre_sp) or _split_numbered_block(pre_af)):
            continue
        k1 = _keyframe_for_task(item_dir, s0, prefer_j=1, require_image=bool(attach_evidence))
        if bool(attach_evidence) and not k1:
            continue
        img = k1[2] if k1 else ""
        q = (
            f'High-level goal: "{hl}" Previous step goal: "{sg0}" Next step goal: "{sg1}" '
            "How does the previous step's outcome support the next step by satisfying its preconditions?"
        )
        eff_cands = [p for p in (_split_numbered_block(eff_sp) + _split_numbered_block(eff_af)) if p]
        pre_cands = [p for p in (_split_numbered_block(pre_sp) + _split_numbered_block(pre_af)) if p]
        if not eff_cands or not pre_cands:
            continue

        best_eff = eff_cands[0]
        best_pre = pre_cands[0]
        best_score = -1
        for e in eff_cands:
            te = _normalize_terms(e)
            if not te:
                continue
            for p in pre_cands:
                sc = len(te & _normalize_terms(p))
                if sc > best_score:
                    best_score = sc
                    best_eff = e
                    best_pre = p
        if best_score <= 0:
            best_eff = eff_cands[0]
            best_pre = pre_cands[0]

        eff_clause = _lowercase_first_alpha(_inline_clause(_annotate_observability(best_eff)))
        pre_clause = _lowercase_first_alpha(_inline_clause(_annotate_observability(best_pre)))
        if not eff_clause or not pre_clause:
            continue
        a = _sanitize_space(
            f'Completing "{sg0}" establishes {eff_clause}, thereby supporting "{sg1}" because it helps ensure {pre_clause}.'
        )
        out.append(
            Sample(
                TASK_14,
                EVIDENCE_KEYFRAME,
                _img_list(img),
                None,
                q,
                a,
                source_rel,
                llm_fields={
                    "high_level_goal": hl,
                    "prev_step_goal": sg0,
                    "prev_step_effects": {"spatial": eff_sp, "affordance": eff_af},
                    "next_step_goal": sg1,
                    "next_step_preconditions": {"spatial": pre_sp, "affordance": pre_af},
                },
            )
        )
    return out


def _make_task18(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    for i in range(len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]
        sid0 = int(s0.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue
        video_rel: Optional[str] = None
        if bool(attach_evidence):
            video = _resolve_video_prefix_or_step_clip(
                item_dir,
                sid0,
                ffmpeg_bin=ffmpeg_bin,
                build=build_video_prefix_clips,
                overwrite=overwrite_video_prefix_clips,
                strict=bool(strict_prefix_video),
            )
            if not video:
                continue
            video_rel = _safe_relpath(video, input_root)
        q = f'Context: High-level goal: "{hl}" Last completed step (in this prefix): "{sg0}" What step goal should come next?'
        evidence_type = EVIDENCE_PREFIX
        images = []
        out.append(
            Sample(
                TASK_15,
                evidence_type,
                images,
                video_rel,
                q,
                sg1,
                source_rel,
                llm_fields={
                    "high_level_goal": hl,
                    "last_completed_step_goal": sg0,
                    "next_step_goal": sg1,
                    "prefix_end_step_id": sid0,
                },
            )
        )
    return out


def _make_task19(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    head: int,
    tail: int,
    *,
    attach_evidence: bool,
) -> Optional[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 3 or not hl:
        return None
    middle = [str(s.get("step_goal", "")).strip() for s in steps[1:-1] if str(s.get("step_goal", "")).strip()]
    if not middle:
        return None
    a = "The missing middle step goals, in order, are: " + " ".join([f'{i+1}) \"{sg}\"' for i, sg in enumerate(middle)])
    q = f'High-level goal: "{hl}" Given glimpses from the beginning and end of the video, infer the missing middle step goals in order.'
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    images: List[str] = []
    if bool(attach_evidence):
        sampled = _list_sampled_frames(item_dir)
        imgs = _pick_head_tail(sampled, head=head, tail=tail)
        if len(imgs) < 2:
            return None
        images = [_safe_relpath(p, input_root) for p in imgs]
    return Sample(
        task_name=TASK_16,
        evidence_type=EVIDENCE_UNIFORM,
        image=images,
        video=None,
        question=q,
        answer=_sanitize_space(a),
        source_path=source_rel,
        llm_fields={"high_level_goal": hl, "middle_step_goals": middle},
    )


def _make_task20(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    rng: random.Random,
    *,
    attach_evidence: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2 or not hl:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)


    max_i = min(3, len(steps) - 1)
    i_idx = 0 if max_i <= 1 else rng.randrange(0, max_i)
    prefix_step = steps[i_idx]
    prefix_end_step = int(prefix_step.get("step_id", 0) or 0)
    if prefix_end_step <= 0:
        return []
    remaining = steps[i_idx + 1 :]
    if len(remaining) < 1:
        return []
    k = min(max(3, min(6, len(remaining))), len(remaining))
    gold = [str(s.get("step_goal", "")).strip() for s in remaining[:k] if str(s.get("step_goal", "")).strip()]
    if len(gold) < 1:
        return []
    video_rel: Optional[str] = None
    if bool(attach_evidence):
        video = _resolve_video_prefix_or_step_clip(
            item_dir,
            prefix_end_step,
            ffmpeg_bin=ffmpeg_bin,
            build=build_video_prefix_clips,
            overwrite=overwrite_video_prefix_clips,
            strict=bool(strict_prefix_video),
        )
        if not video:
            return []
        video_rel = _safe_relpath(video, input_root)
    last_completed_goal = str(prefix_step.get("step_goal", "")).strip()
    answer = " ".join([f"{j+1}) {sg}" for j, sg in enumerate(gold)])

    images: List[str] = []
    q20 = (
        f'Context: High-level goal: "{hl}" Last completed step (in this prefix): "{last_completed_goal}" '
        f"Based on this prefix, predict the next K={len(gold)} step goals, in order."
    )
    out.append(
        Sample(
            TASK_17,
            EVIDENCE_PREFIX,
            images,
            video_rel,
            q20,
            _sanitize_space(answer),
            source_rel,
            llm_fields={
                "high_level_goal": hl,
                "last_completed_step_goal": last_completed_goal,
                "next_step_goals": gold,
                "k": len(gold),
                "prefix_end_step_id": prefix_end_step,
            },
        )
    )
    return out


def _make_task21_bad_plan_diagnosis_and_repair(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    rng: random.Random,
    *,
    attach_evidence: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 4 or not hl:
        return []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)


    max_prefix_idx = min(2, len(steps) - 4)
    if max_prefix_idx < 0:
        return []
    i_idx = rng.randrange(0, max_prefix_idx + 1)
    prefix_step = steps[i_idx]
    prefix_end_step = int(prefix_step.get("step_id", 0) or 0)
    if prefix_end_step <= 0:
        return []

    remaining = steps[i_idx + 1 :]
    if len(remaining) < 3:
        return []
    k = min(3, len(remaining))
    gold = [str(s.get("step_goal", "")).strip() for s in remaining[:k] if str(s.get("step_goal", "")).strip()]
    if len(gold) != k:
        return []


    bad = list(gold)
    flaw_pos = rng.randrange(0, len(bad))

    later_pool = [str(s.get("step_goal", "")).strip() for s in remaining[k:] if str(s.get("step_goal", "")).strip()]
    earlier_pool = [str(s.get("step_goal", "")).strip() for s in steps[: i_idx + 1] if str(s.get("step_goal", "")).strip()]

    def _uniq_pool(xs: List[str]) -> List[str]:
        out_x: List[str] = []
        for x in xs:
            x2 = str(x or "").strip()
            if not x2 or x2 == gold[flaw_pos] or x2 in out_x:
                continue
            out_x.append(x2)
        return out_x

    later_pool_u = _uniq_pool(later_pool)
    earlier_pool_u = _uniq_pool(earlier_pool)
    dup_pool_u = _uniq_pool([g for g in gold if g])
    goal_inconsistent_pool_u = _uniq_pool(
        [
            "Leave the workspace and stop.",
            "Stop and end the process.",
            "Stop and do nothing further.",
        ]
    )

    options: List[Tuple[str, List[str]]] = []
    if later_pool_u:
        options.append(("precondition_missing", later_pool_u))
    if earlier_pool_u:
        options.append(("redundant_step", earlier_pool_u))
    if dup_pool_u:
        options.append(("redundant_step", dup_pool_u))
    if goal_inconsistent_pool_u:
        options.append(("goal_inconsistent", goal_inconsistent_pool_u))
    if not options:
        return []

    flaw_type, pool = options[rng.randrange(0, len(options))]
    bad[flaw_pos] = pool[rng.randrange(0, len(pool))]
    video_rel: Optional[str] = None
    if bool(attach_evidence):
        video = _resolve_video_prefix_or_step_clip(
            item_dir,
            prefix_end_step,
            ffmpeg_bin=ffmpeg_bin,
            build=build_video_prefix_clips,
            overwrite=overwrite_video_prefix_clips,
            strict=bool(strict_prefix_video),
        )
        if not video:
            return []
        video_rel = _safe_relpath(video, input_root)

    bad_steps_inline = " ".join([f'{i+1}) "{s}"' for i, s in enumerate(bad)])
    q = (
        f'Context: High-level goal: "{hl}" Based on this prefix, the following bad_plan_steps are proposed as the next steps: '
        f"{bad_steps_inline} Identify the flaw and repair the plan. "
        'Output in the format: FlawStep=...; FlawType=...; Reason=...; Repair: 1) "..." 2) "..." 3) "..."'
    )
    reason = ""
    if flaw_type == "goal_inconsistent":
        reason = "This step prematurely terminates progress toward the high-level goal, so the remaining necessary steps cannot be completed"
    elif flaw_type == "redundant_step":
        reason = "This step is redundant in the current context, so it wastes a planning slot and causes a required next action to be omitted"
    else:
        reason = "This step is pulled from later in the plan and is out of order, so required intermediate preconditions are likely not satisfied yet"
    repair_inline = " ".join([f'{i+1}) "{s}"' for i, s in enumerate(gold)])
    a = f"FlawStep={flaw_pos+1}; FlawType={flaw_type}; Reason={reason}; Repair: {repair_inline}"

    return [
        Sample(
            TASK_18,
            EVIDENCE_PREFIX,
            [],
            video_rel,
            q,
            a,
            source_rel,
            llm_fields={
                "high_level_goal": hl,
                "bad_plan_steps": list(bad),
                "repair_steps": list(gold),
                "flaw_step": flaw_pos + 1,
                "flaw_type": flaw_type,
                "prefix_end_step_id": prefix_end_step,
            },
        )
    ]


def _counterfactual_clause(question: str) -> str:
    s = str(question or "").strip()
    if not s:
        return ""
    s = re.sub(r"^\s*what\s+if\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" ?!.").strip()
    if not s:
        return ""
    if s[:1].isupper():
        s = s[:1].lower() + s[1:]
    return s


def _make_task22_24(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    out: List[Sample] = []

    def _img_list(path: str) -> List[str]:
        if not bool(attach_evidence):
            return []
        p = str(path or "").strip()
        if not p:
            return []
        return [_safe_relpath(p, input_root)]

    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue
        k0 = _keyframe_for_task(item_dir, st, prefer_j=0, require_image=bool(attach_evidence))
        if bool(attach_evidence) and not k0:
            continue
        img = k0[2] if k0 else ""
        img_rel = _img_list(img)

        q_cf = _require_str(st, "counterfactual_challenge_question")
        a_cf = _require_str(st, "expected_challenge_outcome")
        if q_cf and a_cf:
            q_cf_inline = str(q_cf).strip()
            clause = _counterfactual_clause(q_cf_inline) or q_cf_inline.rstrip("?")
            q22 = (
                f'Context: Step goal: "{step_goal}" What is the most likely outcome if {clause}? '
                "Give a short, physically grounded outcome prediction (spatial setup + affordance/mechanism), and do not propose any recovery actions."
            )
            cleaned = _clean_counterfactual_outcome(a_cf)
            if cleaned:
                out.append(
                    Sample(
                        TASK_19,
                        EVIDENCE_KEYFRAME,
                        img_rel,
                        None,
                        q22,
                        cleaned,
                        source_rel,
                        llm_fields={
                            "step_goal": step_goal,
                            "counterfactual_challenge_question": q_cf_inline,
                            "expected_challenge_outcome": a_cf,
                            "expected_outcome": cleaned,
                        },
                    )
                )

        fr = st.get("failure_reflecting") if isinstance(st.get("failure_reflecting"), dict) else {}
        reason = _require_str(fr, "reason") if isinstance(fr, dict) else ""
        strat = _require_str(fr, "recovery_strategy") if isinstance(fr, dict) else ""
        if reason and strat:
            q25 = (
                f'Context: Step goal: "{step_goal}" Failure reason: "{reason}" '
                "What recovery strategy would most plausibly work? Briefly justify it using spatial stability and affordance/mechanism."
            )
            strategy = strat.strip()
            if strategy and not strategy.endswith((".", "!", "?")):
                strategy = strategy + "."
            a25 = strategy
            reason_clause = reason.strip().rstrip(".!?").strip()
            reason_clause = _lowercase_first_alpha(reason_clause)
            if reason_clause:
                a25 = _sanitize_space(
                    f"{strategy} This improves spatial stability and the relevant affordance/mechanism by addressing the failure mode that {reason_clause}."
                )
            out.append(
                Sample(
                    TASK_20,
                    EVIDENCE_KEYFRAME,
                    img_rel,
                    None,
                    q25,
                    a25,
                    source_rel,
                    llm_fields={"step_goal": step_goal, "failure_reason": reason, "recovery_strategy": strat},
                )
            )

            if sid <= 1:
                continue
            video_rel: Optional[str] = None
            if bool(attach_evidence):
                video = _resolve_video_prefix_or_step_clip(
                    item_dir,
                    sid - 1,
                    ffmpeg_bin=ffmpeg_bin,
                    build=build_video_prefix_clips,
                    overwrite=overwrite_video_prefix_clips,
                    strict=bool(strict_prefix_video),
                )
                if not video:
                    continue
                video_rel = _safe_relpath(video, input_root)
            q27 = (
                f'Context: High-level goal: "{hl}" Failure reason: "{reason}" Recovery strategy: "{strat}" '
                "After applying the recovery strategy, what step goal should come next? Answer as a single step_goal."
            )

            out.append(
                Sample(
                    INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK,
                    EVIDENCE_PREFIX,
                    [],
                    video_rel,
                    q27,
                    step_goal,
                    source_rel,
                    llm_fields={
                        "high_level_goal": hl,
                        "failure_reason": reason,
                        "recovery_strategy": strat,
                        "next_step_goal": step_goal,
                    },
                )
            )

    return out


def generate_samples_for_item(
    *,
    item_dir: str,
    input_root: str,
    enabled_tasks: set[str],
    uniform_k: int,
    head: int,
    tail: int,
    require_videos: bool,
    attach_evidence: bool,
    ffmpeg_bin: str,
    build_video_prefix_clips: bool,
    overwrite_video_prefix_clips: bool,
    strict_prefix_video: bool,
    strict_schema: bool,
    min_steps: int,
    rng: random.Random,
) -> List[Sample]:
    plan_path = os.path.join(item_dir, "causal_plan_with_keyframes.json")
    plan = _read_json(plan_path)
    if strict_schema:
        source_rel = _safe_relpath(plan_path, input_root)
        try:
            _validate_final_plan_schema(plan, source=source_rel, strict=True)
        except ValueError as e:


            if "must be a single snake_case token" in str(e):
                changed = _fix_final_schema_identifiers_in_plan(plan)
                if changed:
                    logger.warning(f"Schema identifiers normalized (in-memory) for {source_rel}; retrying strict validation.")
                    _validate_final_plan_schema(plan, source=source_rel, strict=True)
                else:
                    raise
            else:
                raise

    if int(min_steps) > 0:
        steps = _sorted_steps(plan)
        if len(steps) < int(min_steps):
            raise ValueError(f"Too few steps for generation: steps={len(steps)} < min_steps={int(min_steps)}")

    out: List[Sample] = []

    if TASK_08 in enabled_tasks:
        s = _make_task01(item_dir, plan, input_root, uniform_k=uniform_k, attach_evidence=bool(attach_evidence))
        if s:
            out.append(s)
    if TASK_09 in enabled_tasks:
        s = _make_task02(item_dir, plan, input_root, uniform_k=uniform_k, rng=rng, attach_evidence=bool(attach_evidence))
        if s:
            out.append(s)
    if TASK_10 in enabled_tasks:
        out.extend(_make_task03(item_dir, plan, input_root, require_video=require_videos, attach_evidence=bool(attach_evidence)))
    if any(
        t in enabled_tasks
        for t in (
            INTERNAL_PATIENT_IDENTIFICATION_TASK,
            TASK_11,
            INTERNAL_HOTSPOT_AFFORDANCE_TYPE_TASK,
            INTERNAL_HOTSPOT_MECHANISM_TASK,
            TASK_04,
            TASK_12,
            TASK_05,
            TASK_13,
            TASK_01,
            TASK_02,
            TASK_03,
            TASK_06,
            TASK_07,
        )
    ):
        out.extend(
            _make_task04_to_16(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
            )
        )
        out = [s for s in out if s.task_name in enabled_tasks]
    if TASK_14 in enabled_tasks:
        out.extend(_make_task17(item_dir, plan, input_root, attach_evidence=bool(attach_evidence)))
    if TASK_15 in enabled_tasks:
        out.extend(
            _make_task18(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
                ffmpeg_bin=ffmpeg_bin,
                build_video_prefix_clips=build_video_prefix_clips,
                overwrite_video_prefix_clips=overwrite_video_prefix_clips,
                strict_prefix_video=bool(strict_prefix_video),
            )
        )
    if TASK_16 in enabled_tasks:
        s = _make_task19(item_dir, plan, input_root, head=head, tail=tail, attach_evidence=bool(attach_evidence))
        if s:
            out.append(s)
    if TASK_17 in enabled_tasks:
        out.extend(
            _make_task20(
                item_dir,
                plan,
                input_root,
                rng=rng,
                attach_evidence=bool(attach_evidence),
                ffmpeg_bin=ffmpeg_bin,
                build_video_prefix_clips=build_video_prefix_clips,
                overwrite_video_prefix_clips=overwrite_video_prefix_clips,
                strict_prefix_video=bool(strict_prefix_video),
            )
        )
    if TASK_18 in enabled_tasks:
        out.extend(
            _make_task21_bad_plan_diagnosis_and_repair(
                item_dir,
                plan,
                input_root,
                rng=rng,
                attach_evidence=bool(attach_evidence),
                ffmpeg_bin=ffmpeg_bin,
                build_video_prefix_clips=build_video_prefix_clips,
                overwrite_video_prefix_clips=overwrite_video_prefix_clips,
                strict_prefix_video=bool(strict_prefix_video),
            )
        )
    if any(t in enabled_tasks for t in (TASK_19, TASK_20, INTERNAL_NEXT_STEP_AFTER_RECOVERY_TASK)):
        out.extend(
            _make_task22_24(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
                ffmpeg_bin=ffmpeg_bin,
                build_video_prefix_clips=build_video_prefix_clips,
                overwrite_video_prefix_clips=overwrite_video_prefix_clips,
                strict_prefix_video=bool(strict_prefix_video),
            )
        )


    final: List[Sample] = []
    for s in out:
        if s.task_name not in enabled_tasks:
            continue
        images = [] if not bool(attach_evidence) else [p for p in s.image if p]
        video = None
        if bool(attach_evidence):
            video = s.video.strip() if isinstance(s.video, str) and s.video.strip() else None
            if not images and not video:
                continue
        if _has_frame_leak(s.question) or _has_frame_leak(s.answer):
            continue
        answer = _sanitize_space(s.answer)
        if s.task_name in SINGLE_SENTENCE_TASKS:
            answer = _enforce_single_sentence(answer)
        final.append(
            Sample(
                task_name=s.task_name,
                evidence_type=s.evidence_type,
                image=images,
                video=video,
                question=_sanitize_space(s.question),
                answer=answer,
                source_path=s.source_path,
                llm_fields=s.llm_fields,
            )
        )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Mani-LongVideo QA dataset (two-stage single-step subset: Task_08–Task_07 + Task_19–Task_20) "
            "from causal_plan_with_keyframes.json (final schema)."
        )
    )
    parser.add_argument("--input-root", required=True, help="Dataset root containing many item dirs with causal_plan_with_keyframes.json.")
    parser.add_argument("--output-dir", required=True, help="Output root directory (will create one folder per task with data.jsonl).")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--keep-going",
        dest="keep_going",
        action="store_true",
        help="Never abort generation due to item-level errors/audit/require checks; always save outputs and write issues JSON (default).",
    )
    run_mode.add_argument(
        "--fail-fast",
        dest="keep_going",
        action="store_false",
        help="Abort on errors / return non-zero exit codes (strict-failure behavior).",
    )
    parser.set_defaults(keep_going=True)
    parser.add_argument(
        "--issues-prefix",
        default="issues",
        help="Prefix for issue logs under output_dir (writes <prefix>_errors.jsonl/json and <prefix>_warnings.jsonl/json).",
    )
    out_mode = parser.add_mutually_exclusive_group()
    out_mode.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output_dir task files if they already exist (only use this when intentionally merging runs).",
    )
    out_mode.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete output_dir before writing new outputs (destructive).",
    )
    out_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing output_dir (skip items already recorded in resume_state.jsonl).",
    )
    parser.add_argument(
        "--resume-allow-mismatch",
        action="store_true",
        help="Allow --resume even if run_config.json differs (may mix settings in one output_dir).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Write text-only QA (omit image/video paths and meta.evidence_files); also skips evidence existence checks in --audit.",
    )
    parser.add_argument(
        "--meta-abs-paths",
        action="store_true",
        help="Also store absolute paths in meta (meta.item_dir_abs/meta.source_path_abs/meta.input_root_abs) for easier later alignment; may reduce portability.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N item dirs (0 = no limit).")
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Fail if an item has fewer than N steps (two-stage data is typically single-step, so N=1 is common).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(DEFAULT_TASKS),
        help="Subset of task names to generate (default: canonical two-stage single-step task subset).",
    )
    parser.add_argument("--uniform-k", type=int, default=8, help="Number of uniform frames for images_uniform_scene tasks (default: 8).")
    parser.add_argument("--head", type=int, default=4, help="Head frames for Task_16 (default: 4).")
    parser.add_argument("--tail", type=int, default=4, help="Tail frames for Task_16 (default: 4).")
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="If set, require video_clip assets for Task_10 (skip keyframe fallback). Ignored in --text-only mode.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable path (used by --build-video-prefix-clips).",
    )
    parser.add_argument(
        "--build-video-prefix-clips",
        action="store_true",
        help=(
            "If set, build missing video_prefix assets under <item_dir>/cumulative_last_frame_segments/ "
            "by concatenating step clips via ffmpeg (kept for CLI compatibility; typically unused for two-stage single-step items)."
        ),
    )
    parser.add_argument(
        "--overwrite-video-prefix-clips",
        action="store_true",
        help="If set, overwrite existing video_prefix assets when --build-video-prefix-clips is enabled.",
    )
    parser.add_argument(
        "--strict-prefix-video",
        action="store_true",
        help="Require true cumulative prefix clips for video_prefix tasks; do not fall back to the last-step clip.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--strict-schema",
        dest="strict_schema",
        action="store_true",
        help="Enable strict final-schema validation for causal_plan_with_keyframes.json (default).",
    )
    schema_group.add_argument(
        "--no-strict-schema",
        dest="strict_schema",
        action="store_false",
        help="Disable strict schema validation (not recommended for final generation).",
    )
    parser.set_defaults(strict_schema=True)
    parser.add_argument("--no-api", action="store_true", help="Disable OpenAI-compatible API two-stage rewriting; keep deterministic answers.")
    parser.add_argument(
        "--llm-tasks",
        nargs="*",
        default=list(DEFAULT_LLM_TASKS),
        help="Tasks to rewrite/polish via API (default: all tasks).",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=0, help="Override MAX_TOKENS for API calls (0 uses env/default).")
    parser.add_argument("--llm-temperature", type=float, default=0.3, help="Override TEMPERATURE for API calls (default: 0.3).")
    parser.add_argument("--llm-single-pass", action="store_true", help="Use a single API pass (no second polishing pass).")
    parser.add_argument(
        "--llm-require-success",
        action="store_true",
        help=(
            "If set, treat any LLM call failure as an item-level error (prevents silently falling back to deterministic drafts). "
            "Recommended for high-stakes large-scale generation."
        ),
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail fast if LLM polishing is requested but the API client is not available (missing API_KEY, etc.).",
    )
    parser.add_argument(
        "--llm-fallback",
        choices=["draft", "skip", "fail"],
        default="draft",
        help="If the LLM output is rejected by safety checks, use draft (default), skip the sample, or fail the run.",
    )
    parser.add_argument(
        "--llm-vocab-guard",
        choices=["strict", "warn", "off"],
        default="off",
        help=(
            "Novel-terms vocabulary guard for LLM outputs. "
            "strict=reject when non-trivial terms are not present in the source JSON/draft; "
            "warn=allow but report in logs; off=disable this check."
        ),
    )
    parser.add_argument(
        "--llm-verify",
        choices=["strict", "warn", "off"],
        default="warn",
        help=(
            "LLM output verification mode. "
            "strict=reject on any grounding/format drift; "
            "warn=log drift but keep the output (default; recommended for large-scale runs); "
            "off=disable most checks (still rejects empty/prompt-echo/frame-leak/Task_18 format)."
        ),
    )
    parser.add_argument(
        "--audit",
        dest="audit",
        action="store_true",
        default=True,
        help=(
            "Run built-in QA audit on output_dir after generation (default: on). "
            "Errors are logged; use --audit-strict to make audit errors fail the run."
        ),
    )
    parser.add_argument(
        "--no-audit",
        dest="audit",
        action="store_false",
        help="Disable built-in QA audit on output_dir after generation.",
    )
    parser.add_argument(
        "--audit-deep",
        dest="audit_deep",
        action="store_true",
        default=True,
        help=(
            "Run deep grounding audit (default: on; verifies Q/A against meta.source_path JSON fields; no evidence checks). "
            "Errors are logged; use --audit-strict to make deep-audit errors fail the run."
        ),
    )
    parser.add_argument(
        "--no-audit-deep",
        dest="audit_deep",
        action="store_false",
        help="Disable deep grounding audit on output_dir after generation.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run built-in QA audit on output_dir; do not generate new samples.",
    )
    parser.add_argument(
        "--audit-max-issues",
        type=int,
        default=50,
        help="Maximum number of audit issues to print (default: 50).",
    )
    parser.add_argument(
        "--audit-strict",
        action="store_true",
        help="If set, treat audit/deep-audit errors as fatal (exit code 2). Default: log errors but continue.",
    )
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help="Fail if any enabled task produces 0 samples overall.",
    )
    parser.add_argument(
        "--require-all-tasks-per-item",
        action="store_true",
        help=(
            "Fail if any item does not produce at least one sample for every enabled task. "
            "Incomplete items are reported in output_dir/incomplete_items.json (samples are still written)."
        ),
    )
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir)
    issues_prefix = str(args.issues_prefix or "issues").strip() or "issues"
    keep_going = bool(args.keep_going)
    issue_paths = _issues_paths(output_dir=output_dir, prefix=issues_prefix)
    try:
        os.makedirs(output_dir, exist_ok=True)
        open(issue_paths["errors_jsonl"], "a", encoding="utf-8").close()
        open(issue_paths["warnings_jsonl"], "a", encoding="utf-8").close()
    except Exception:
        pass
    def _atexit_finalize_issues() -> None:
        try:
            _finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
        except Exception:
            pass
    atexit.register(_atexit_finalize_issues)

    if bool(args.audit_only):
        audit_only_exit_code = 0

        def _audit_issue_sink(iss: AuditIssue) -> None:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        def _deep_audit_issue_sink(iss: AuditIssue) -> None:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="deep_audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        try:
            report = _audit_output_dir(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                require_evidence=not bool(args.text_only),
                issue_sink=_audit_issue_sink,
            )
        except Exception as e:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_only",
                message=f"Audit failed: {type(e).__name__}: {e}",
                exc=e,
            )
            if not keep_going:
                audit_only_exit_code = max(audit_only_exit_code, 2)
            report = AuditReport(total_samples=0, errors=1, warnings=0, issues=[])

        logger.info(f"Audit summary: samples={report.total_samples} errors={report.errors} warnings={report.warnings}")
        try:
            with open(os.path.join(output_dir, "audit_report.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        for iss in report.issues:
            log_fn = logger.error if iss.severity == "error" else logger.warning
            log_fn(f"AUDIT[{iss.severity}] task={iss.task_name} file={iss.file} line={iss.line} id={iss.sample_id}: {iss.message}")
        total_issues = int(report.errors) + int(report.warnings)
        if total_issues > len(report.issues):
            logger.info(f"Audit issues truncated: showing {len(report.issues)}/{total_issues} (increase --audit-max-issues).")
        if report.errors and bool(args.audit_strict):
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_strict",
                message="Audit reported errors and --audit-strict is set.",
            )
            if not keep_going:
                audit_only_exit_code = max(audit_only_exit_code, 2)
            logger.warning("Audit reported errors but keep-going is enabled; continuing.")
        if report.errors and not bool(args.audit_strict):
            logger.warning("Audit reported errors. --audit-strict is not set; continuing.")

        if bool(args.audit_deep):
            try:
                deep = _audit_output_dir_deep(
                    output_dir=output_dir,
                    input_root=input_root,
                    max_issues=int(args.audit_max_issues),
                    issue_sink=_deep_audit_issue_sink,
                )
            except Exception as e:
                _log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="audit_only_deep",
                    message=f"Deep audit failed: {type(e).__name__}: {e}",
                    exc=e,
                )
                if not keep_going:
                    audit_only_exit_code = max(audit_only_exit_code, 2)
                deep = AuditReport(total_samples=0, errors=1, warnings=0, issues=[])
            logger.info(f"Deep audit summary: samples={deep.total_samples} errors={deep.errors} warnings={deep.warnings}")
            try:
                with open(os.path.join(output_dir, "deep_audit_report.json"), "w", encoding="utf-8") as f:
                    json.dump(asdict(deep), f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            for iss in deep.issues:
                log_fn = logger.error if iss.severity == "error" else logger.warning
                log_fn(f"DEEP_AUDIT[{iss.severity}] task={iss.task_name} file={iss.file} line={iss.line} id={iss.sample_id}: {iss.message}")
            total_issues = int(deep.errors) + int(deep.warnings)
            if total_issues > len(deep.issues):
                logger.info(f"Deep audit issues truncated: showing {len(deep.issues)}/{total_issues} (increase --audit-max-issues).")
            if deep.errors and bool(args.audit_strict):
                _log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="audit_strict",
                    message="Deep audit reported errors and --audit-strict is set.",
                )
                if not keep_going:
                    audit_only_exit_code = max(audit_only_exit_code, 2)
                logger.warning("Deep audit reported errors but keep-going is enabled; continuing.")
            if deep.errors and not bool(args.audit_strict):
                logger.warning("Deep audit reported errors. --audit-strict is not set; continuing.")

        try:
            _finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
        except Exception:
            pass
        if audit_only_exit_code and not keep_going:
            raise SystemExit(audit_only_exit_code)
        return

    existing = sorted(glob.glob(os.path.join(output_dir, "*", "data.jsonl")))
    if existing:
        if bool(args.overwrite_output_dir):
            shutil.rmtree(output_dir)
        elif not (bool(args.append) or bool(args.resume)):
            raise SystemExit(
                f"Refusing to write to a non-empty output_dir (found {len(existing)} existing task data.jsonl). "
                "Use a fresh --output-dir, or pass --resume to continue from a previous run, or pass --append to merge runs, "
                "or --overwrite-output-dir to delete it."
            )

    if bool(args.build_video_prefix_clips) and not _ffmpeg_exists(str(args.ffmpeg_bin)):
        logger.warning(
            f"ffmpeg not found for --ffmpeg-bin={args.ffmpeg_bin!r}; clip building flags will not work. "
            "Install ffmpeg or set --ffmpeg-bin, or rely on the script's video fallbacks."
        )
        if bool(args.strict_prefix_video):
            logger.warning("Strict prefix-video flag is enabled; some tasks may be skipped without prebuilt prefix clips.")

    enabled_tasks = set(args.tasks or [])
    llm_tasks = set(args.llm_tasks or [])

    attach_evidence = not bool(args.text_only)

    unknown = sorted([t for t in enabled_tasks if t not in set(ALL_TASKS)])
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    unknown_llm = sorted([t for t in llm_tasks if t not in set(ALL_TASKS)])
    if unknown_llm:
        raise ValueError(f"Unknown llm task names: {unknown_llm}")

    item_dirs = _list_item_dirs(input_root)
    if args.limit and int(args.limit) > 0:
        item_dirs = item_dirs[: int(args.limit)]
    if not item_dirs:
        raise FileNotFoundError(f"No item dirs found under {input_root} (expecting causal_plan_with_keyframes.json).")

    logger.info(f"Found {len(item_dirs)} item dirs under: {input_root}")
    logger.info(f"Enabled tasks: {sorted(enabled_tasks)}")
    if llm_tasks:
        logger.info(f"LLM rewrite tasks: {sorted(llm_tasks)}")

    llm: Optional[TwoStageLlm] = None
    if not bool(args.no_api):
        cfg = ApiConfig()
        if int(args.llm_max_tokens) > 0:
            cfg.max_tokens = int(args.llm_max_tokens)
        cfg.temperature = float(args.llm_temperature)
        llm = TwoStageLlm(cfg)
        if not llm.enabled():
            llm = None
            logger.info("LLM disabled (missing API_KEY or client init failure); proceeding without API rewriting.")
    if bool(args.require_llm):
        if bool(args.no_api):
            msg = "--require-llm is set but --no-api disables LLM polishing."
            _log_issue(output_dir=output_dir, prefix=issues_prefix, severity="error", phase="require_llm", message=msg)
            if not keep_going:
                raise SystemExit(msg)
            logger.warning(msg + " keep-going is enabled; proceeding without LLM polishing.")
            llm = None
        elif llm is None or not llm.enabled():
            msg = "--require-llm is set but the LLM client is not enabled (check API_KEY/API_BASE_URL/MODEL_NAME)."
            _log_issue(output_dir=output_dir, prefix=issues_prefix, severity="error", phase="require_llm", message=msg)
            if not keep_going:
                raise SystemExit(msg)
            logger.warning(msg + " keep-going is enabled; proceeding without LLM polishing.")
            llm = None


    run_config = _build_taskslist_run_config(
        input_root=input_root,
        enabled_tasks=sorted(enabled_tasks),
        text_only=bool(args.text_only),
        meta_abs_paths=bool(args.meta_abs_paths),
        uniform_k=int(args.uniform_k),
        head=int(args.head),
        tail=int(args.tail),
        require_videos=bool(args.require_videos),
        ffmpeg_bin=str(args.ffmpeg_bin),
        build_video_prefix_clips=bool(args.build_video_prefix_clips),
        overwrite_video_prefix_clips=bool(args.overwrite_video_prefix_clips),
        strict_prefix_video=bool(args.strict_prefix_video),
        strict_schema=bool(args.strict_schema),
        min_steps=int(args.min_steps),
        llm_enabled=bool(llm is not None and llm.enabled()),
        llm_tasks=sorted(llm_tasks),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_temperature=float(args.llm_temperature),
        llm_single_pass=bool(args.llm_single_pass),
        llm_vocab_guard=str(args.llm_vocab_guard),
        llm_verify=str(args.llm_verify),
        llm_fallback=str(args.llm_fallback),
        llm_require_success=bool(args.llm_require_success),
    )
    cfg_path = _run_config_path(output_dir)
    prev_cfg = _load_json_maybe(cfg_path)
    if prev_cfg is None:
        if not existing or bool(args.resume):
            if bool(args.resume) and existing and not bool(args.overwrite_output_dir):
                logger.warning(f"--resume is set but missing {cfg_path}; creating it from current args (cannot verify existing outputs).")
            _write_json_atomic(cfg_path, run_config)
        elif bool(args.append):
            logger.warning(f"--append is set but missing {cfg_path}; not writing run_config.json for a merged output_dir.")
    else:
        if bool(args.resume) and not bool(args.resume_allow_mismatch):
            prev_cfg_dict = prev_cfg if isinstance(prev_cfg, dict) else {}
            mism = _run_config_mismatch_keys(prev_cfg_dict, run_config)
            if mism:
                details = "; ".join([f"{k}: old={prev_cfg_dict.get(k)!r} new={run_config.get(k)!r}" for k in mism[:12]])
                more = "" if len(mism) <= 12 else f" (+{len(mism) - 12} more)"
                raise SystemExit(
                    "--resume refused due to run_config.json mismatch. "
                    + details
                    + more
                    + ". Use a fresh --output-dir, or use --append to intentionally merge runs, or pass --resume-allow-mismatch."
                )

    resume_state = _resume_state_path(output_dir)
    processed_items: set[str] = set()
    if bool(args.resume):
        resume_recs = _load_resume_state(resume_state)
        processed_items = {k for k, rec in resume_recs.items() if isinstance(rec, dict) and rec.get("status") == "done"}

        if not processed_items and not resume_recs and existing and not bool(args.overwrite_output_dir):
            inferred = _infer_processed_items_from_output_dir(output_dir, sorted(enabled_tasks))
            if inferred:
                processed_items = set(inferred)
                logger.info(f"Resume: inferred processed items from existing outputs: {len(processed_items)}")

    total = 0
    per_task_counts: Dict[str, int] = {}
    prefix_fallback_samples = 0
    incomplete_items: List[Dict[str, Any]] = []
    incomplete_count = 0
    skipped_items = 0
    exit_code = 0


    for t in sorted(enabled_tasks):
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        open(os.path.join(output_dir, t, "data.jsonl"), "a", encoding="utf-8").close()

    for idx, item_dir in enumerate(item_dirs, start=1):
        rel_item = _safe_relpath(item_dir, input_root)
        if bool(args.resume) and rel_item in processed_items:
            logger.info(f"[{idx}/{len(item_dirs)}] Resume: skip already processed item: {rel_item}")
            continue
        logger.info(f"[{idx}/{len(item_dirs)}] Processing item: {rel_item}")
        item_rng = random.Random(_stable_int_seed(rel_item))
        try:
            samples = generate_samples_for_item(
                item_dir=item_dir,
                input_root=input_root,
                enabled_tasks=enabled_tasks,
                uniform_k=int(args.uniform_k),
                head=int(args.head),
                tail=int(args.tail),
                require_videos=bool(args.require_videos),
                attach_evidence=bool(attach_evidence),
                ffmpeg_bin=str(args.ffmpeg_bin),
                build_video_prefix_clips=bool(args.build_video_prefix_clips),
                overwrite_video_prefix_clips=bool(args.overwrite_video_prefix_clips),
                strict_prefix_video=bool(args.strict_prefix_video),
                strict_schema=bool(args.strict_schema),
                min_steps=int(args.min_steps),
                rng=item_rng,
            )
        except Exception as e:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="item_generation",
                message=f"Item generation failed: {type(e).__name__}: {e}",
                rel_item=rel_item,
                exc=e,
            )
            skipped_items += 1
            logger.error(f"ITEM_ERROR item={rel_item} error={e}")
            if bool(args.require_all_tasks_per_item):
                incomplete_items.append({"item_dir": rel_item, "missing_tasks": sorted(enabled_tasks), "error": str(e)})
            _append_jsonl(
                resume_state,
                {"rel_item": rel_item, "status": "error", "error": f"{type(e).__name__}: {e}"},
            )
            processed_items.add(rel_item)
            if keep_going or bool(args.require_all_tasks_per_item):
                continue
            raise
        if llm is not None and llm_tasks:
            try:
                samples = _apply_llm(
                    samples,
                    llm,
                    llm_tasks,
                    two_pass=not bool(args.llm_single_pass),
                    fallback=str(args.llm_fallback),
                    require_success=bool(args.llm_require_success),
                    vocab_guard=str(args.llm_vocab_guard),
                    verify=str(args.llm_verify),
                )
            except Exception as e:                    
                logger.warning(f"LLM stage failed for item={rel_item}: {e}; proceeding with deterministic drafts.")
                _log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="warn",
                    phase="llm_stage",
                    message=f"LLM stage failed; using deterministic drafts: {type(e).__name__}: {e}",
                    rel_item=rel_item,
                    exc=e,
                )
        missing: List[str] = []
        if bool(args.require_all_tasks_per_item):
            produced = {s.task_name for s in samples}
            missing = sorted([t for t in enabled_tasks if t not in produced])
            if missing:
                incomplete_items.append({"item_dir": rel_item, "missing_tasks": missing})
                incomplete_count += 1
                logger.warning(f"INCOMPLETE_ITEM item={rel_item} missing_tasks={missing}")
                _log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="missing_tasks_per_item",
                    message=f"Item missing enabled tasks: {missing}",
                    rel_item=rel_item,
                    details={"missing_tasks": missing},
                )
        per_item_counts: Dict[str, int] = {}
        for s in samples:
            per_item_counts[s.task_name] = int(per_item_counts.get(s.task_name, 0)) + 1
            if bool(attach_evidence):
                v = str(s.video or "").replace("\\", "/")
                if s.evidence_type == EVIDENCE_PREFIX and v:
                    if "/cumulative_last_frame_segments/" not in v:
                        prefix_fallback_samples += 1
            entry = _sharegpt_entry(s, attach_evidence=bool(attach_evidence))
            if bool(args.meta_abs_paths):
                meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                meta["input_root_abs"] = os.path.abspath(input_root)
                meta["item_dir_abs"] = os.path.abspath(item_dir)
                meta["source_path_abs"] = _abs_under_root(str(s.source_path or ""), input_root) if str(s.source_path or "").strip() else ""
                entry["meta"] = meta
            out_path = os.path.join(output_dir, s.task_name, "data.jsonl")
            try:
                _write_jsonl(out_path, entry)
            except Exception as e:                    
                _log_issue(
                    output_dir=output_dir,
                    prefix=issues_prefix,
                    severity="error",
                    phase="write_jsonl",
                    message=f"Failed to write sample JSONL: {type(e).__name__}: {e}",
                    rel_item=rel_item,
                    task_name=s.task_name,
                    exc=e,
                    details={"out_path": out_path},
                )
                try:
                    _append_jsonl(
                        os.path.join(output_dir, f"{issues_prefix}_unsaved_samples.jsonl"),
                        {
                            "rel_item": rel_item,
                            "task_name": s.task_name,
                            "error": f"{type(e).__name__}: {e}",
                            "entry": entry,
                        },
                    )
                except Exception:
                    pass
                continue
            total += 1
            per_task_counts[s.task_name] = int(per_task_counts.get(s.task_name, 0)) + 1
        rec: Dict[str, Any] = {
            "rel_item": rel_item,
            "status": "done_incomplete" if missing else "done",
            "samples_written": int(len(samples)),
            "per_task_counts": per_item_counts,
        }
        if missing:
            rec["missing_tasks"] = missing
        _append_jsonl(resume_state, rec)
        processed_items.add(rel_item)
        if idx % 10 == 0:
            logger.info(f"Progress: items={idx}, samples_written={total}")
    if per_task_counts:
        counts_inline = ", ".join([f"{k}={per_task_counts[k]}" for k in sorted(per_task_counts.keys())])
        logger.info(f"Per-task counts: {counts_inline}")
    if bool(attach_evidence) and prefix_fallback_samples:
        logger.info(f"Video fallback summary: video_prefix_non_cumulative={prefix_fallback_samples}")
    logger.info(f"Done. Total samples_written={total}. Output_dir={output_dir}")
    if bool(skipped_items):
        logger.warning(f"Skipped items due to item-level errors: {skipped_items}/{len(item_dirs)}")
    if bool(incomplete_count):
        logger.warning(f"Items with missing tasks: {incomplete_count}/{len(item_dirs)}")
    if incomplete_items:
        report_path = os.path.join(output_dir, "incomplete_items.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(incomplete_items, f, ensure_ascii=False, indent=2)
            logger.warning(f"Wrote incomplete items report: {report_path}")
        except Exception as e:                    
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="write_incomplete_items",
                message=f"Failed to write incomplete_items.json: {type(e).__name__}: {e}",
                exc=e,
                details={"path": report_path},
            )

    missing_overall = sorted([t for t in enabled_tasks if not _jsonl_has_any_entry(os.path.join(output_dir, t, "data.jsonl"))])
    if missing_overall:
        msg = f"Some enabled tasks produced 0 samples: {missing_overall}"
        if bool(args.require_all_tasks):
            logger.error(msg)
            exit_code = max(exit_code, 2)
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="missing_tasks_overall",
                message=msg,
                details={"missing_tasks": missing_overall},
            )
        else:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="missing_tasks_overall",
                message=msg,
                details={"missing_tasks": missing_overall},
            )
        logger.warning(msg)
    if incomplete_items:
        exit_code = max(exit_code, 2)
        _log_issue(
            output_dir=output_dir,
            prefix=issues_prefix,
            severity="error",
            phase="incomplete_items",
            message=f"Incomplete items encountered: {len(incomplete_items)}",
            details={"count": int(len(incomplete_items))},
        )

    if bool(args.audit):
        def _audit_issue_sink(iss: AuditIssue) -> None:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        try:
            report = _audit_output_dir(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                require_evidence=bool(attach_evidence),
                issue_sink=_audit_issue_sink,
            )
        except Exception as e:                    
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit",
                message=f"Audit failed: {type(e).__name__}: {e}",
                exc=e,
            )
            if not keep_going:
                raise
            report = AuditReport(total_samples=0, errors=1, warnings=0, issues=[])

        logger.info(f"Audit summary: samples={report.total_samples} errors={report.errors} warnings={report.warnings}")
        try:
            with open(os.path.join(output_dir, "audit_report.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        except Exception as e:                    
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="write_audit_report",
                message=f"Failed to write audit_report.json: {type(e).__name__}: {e}",
                exc=e,
            )
        for iss in report.issues:
            log_fn = logger.error if iss.severity == "error" else logger.warning
            log_fn(f"AUDIT[{iss.severity}] task={iss.task_name} file={iss.file} line={iss.line} id={iss.sample_id}: {iss.message}")
        total_issues = int(report.errors) + int(report.warnings)
        if total_issues > len(report.issues):
            logger.info(f"Audit issues truncated: showing {len(report.issues)}/{total_issues} (increase --audit-max-issues).")
        if report.errors and bool(args.audit_strict):
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_strict",
                message="Audit reported errors and --audit-strict is set.",
            )
            if not keep_going:
                exit_code = max(exit_code, 2)
            logger.warning("Audit reported errors but keep-going is enabled; continuing.")
        if report.errors and not bool(args.audit_strict):
            logger.warning("Audit reported errors. --audit-strict is not set; continuing.")

    if bool(args.audit_deep):
        def _deep_audit_issue_sink(iss: AuditIssue) -> None:
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity=str(iss.severity),
                phase="deep_audit",
                message=str(iss.message),
                task_name=str(iss.task_name),
                sample_id=str(iss.sample_id),
                details={"file": iss.file, "line": int(iss.line)},
            )

        try:
            deep = _audit_output_dir_deep(
                output_dir=output_dir,
                input_root=input_root,
                max_issues=int(args.audit_max_issues),
                issue_sink=_deep_audit_issue_sink,
            )
        except Exception as e:                    
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="deep_audit",
                message=f"Deep audit failed: {type(e).__name__}: {e}",
                exc=e,
            )
            if not keep_going:
                raise
            deep = AuditReport(total_samples=0, errors=1, warnings=0, issues=[])

        logger.info(f"Deep audit summary: samples={deep.total_samples} errors={deep.errors} warnings={deep.warnings}")
        try:
            with open(os.path.join(output_dir, "deep_audit_report.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(deep), f, ensure_ascii=False, indent=2)
        except Exception as e:                    
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="warn",
                phase="write_deep_audit_report",
                message=f"Failed to write deep_audit_report.json: {type(e).__name__}: {e}",
                exc=e,
            )
        for iss in deep.issues:
            log_fn = logger.error if iss.severity == "error" else logger.warning
            log_fn(f"DEEP_AUDIT[{iss.severity}] task={iss.task_name} file={iss.file} line={iss.line} id={iss.sample_id}: {iss.message}")
        total_issues = int(deep.errors) + int(deep.warnings)
        if total_issues > len(deep.issues):
            logger.info(f"Deep audit issues truncated: showing {len(deep.issues)}/{total_issues} (increase --audit-max-issues).")
        if deep.errors and bool(args.audit_strict):
            _log_issue(
                output_dir=output_dir,
                prefix=issues_prefix,
                severity="error",
                phase="audit_strict",
                message="Deep audit reported errors and --audit-strict is set.",
            )
            if not keep_going:
                exit_code = max(exit_code, 2)
            logger.warning("Deep audit reported errors but keep-going is enabled; continuing.")
        if deep.errors and not bool(args.audit_strict):
            logger.warning("Deep audit reported errors. --audit-strict is not set; continuing.")

    if exit_code:
        if not keep_going:
            try:
                _finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
            except Exception:
                pass
            raise SystemExit(exit_code)
        logger.warning(f"Completed with exit_code={exit_code} but keep-going is enabled; outputs are preserved.")

    try:
        _finalize_issue_json_files(output_dir=output_dir, prefix=issues_prefix)
    except Exception:
        pass


if __name__ == "__main__":
    main()
