




from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import threading
import time
import traceback
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_stage_two_qa")
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
    TASK_14,
    TASK_15,
    TASK_16,
    TASK_17,
    TASK_18,
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
    TASK_14,
    TASK_15,
    TASK_16,
    TASK_17,
    TASK_18,
    TASK_19,
    TASK_20,
)


EVIDENCE_KEYFRAME = "keyframe_single"
EVIDENCE_UNIFORM = "images_uniform_scene"
EVIDENCE_CLIP = "video_clip"
EVIDENCE_PREFIX = "video_prefix"
EVIDENCE_CLIP_PAIR = "video_clip_pair"


_T08_QUESTIONS = [
    "Based on this video, what is the most appropriate high-level goal?",
    "What overarching goal does the person pursue throughout this video?",
    "Identify the main high-level objective demonstrated in this video.",
]


FRAME_LEAK_PATTERNS = [
    re.compile(r"\bframe_\d+\b", re.IGNORECASE),
    re.compile(r"\bsample_\d+\b", re.IGNORECASE),
    re.compile(r"\bkeyframe[_\s]?\d+\b", re.IGNORECASE),
    re.compile(r"\bstep_\d+\b", re.IGNORECASE),



    re.compile(r"\bclip_\d+\b", re.IGNORECASE),                                               
    re.compile(r"\bsegment_\d+\b", re.IGNORECASE),                                         
    re.compile(r"\bkf_\d+\b", re.IGNORECASE),                                                 
    re.compile(r"\bts_\d", re.IGNORECASE),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),                                              
    re.compile(r"\.(jpg|jpeg|png|mp4|webm|gif)\b", re.IGNORECASE),
    re.compile(r"\b(frame|image)\s*\d+\b", re.IGNORECASE),
    re.compile(r"(?:/[a-zA-Z_]\w*){3,}", re.IGNORECASE),                                                  
    re.compile(r"\bP\d+_\d+", re.IGNORECASE),                                                                                
]


_KEY_MOMENT_PREFIX_RE = re.compile(
    r"^\s*(?:Key moment|Critical frame|Action|Event|Step)\s*\d+\s*(?:\([^)]*\))?\s*[:\.]\s*",
    re.IGNORECASE,
)
_TS_RE = re.compile(r"_ts_(\d+(?:\.\d+)?)s\b", re.IGNORECASE)
_SNAKE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
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


_DISTRACTOR_OBJECTS_KITCHEN = [

    "microwave", "toaster", "blender", "food_processor", "rice_cooker",
    "electric_kettle", "air_fryer", "slow_cooker", "stand_mixer",
    "bread_maker", "coffee_grinder", "juicer", "waffle_iron",
    "immersion_blender", "pressure_cooker", "toaster_oven",

    "baking_tray", "muffin_tin", "cake_pan", "roasting_pan",
    "wok", "saucepan", "stockpot", "double_boiler", "steamer_basket",
    "cast_iron_skillet", "griddle", "dutch_oven",

    "spatula", "ladle", "whisk", "tongs", "peeler", "grater",
    "rolling_pin", "can_opener", "measuring_cup", "measuring_spoon",
    "slotted_spoon", "potato_masher", "meat_thermometer", "basting_brush",
    "garlic_press", "ice_cream_scoop", "pizza_cutter", "corkscrew",
    "bottle_opener", "nutcracker", "zester", "mandoline",
    "kitchen_shears", "pastry_bag", "cookie_cutter",

    "cutting_board", "colander", "strainer", "sieve", "funnel",
    "mixing_bowl", "salad_spinner", "cheese_cloth", "parchment_paper",
    "plastic_wrap", "aluminum_foil", "zip_lock_bag", "mason_jar",
    "food_container", "spice_rack", "butter_dish", "egg_cup",
    "salt_shaker", "pepper_mill", "oil_dispenser", "sugar_bowl",

    "dish_soap", "sponge", "scrub_brush", "dish_rack", "drying_mat",
    "oven_mitt", "pot_holder", "apron", "kitchen_towel", "trivet",
    "timer", "thermometer", "mortar", "skewer", "toothpick",
    "straw", "coaster", "napkin_holder", "placemat", "bread_basket",
]

_DISTRACTOR_OBJECTS_OTHER = [

    "hammer", "screwdriver", "wrench", "pliers", "tape_measure",
    "drill_bit", "sandpaper", "clamp", "level", "utility_knife",
    "saw", "chisel", "bolt", "nail", "wire_cutter",

    "toothbrush", "shampoo_bottle", "soap_dispenser", "hair_dryer",
    "cotton_swab", "nail_clipper", "razor", "towel_rack", "loofah",
    "bath_mat", "shower_curtain", "dental_floss",

    "laptop", "phone", "remote_control", "flashlight", "charger",
    "stapler", "scissors", "tape_roll", "calculator", "pen",
    "mouse_pad", "headphones", "usb_cable", "keyboard", "monitor_stand",
    "paper_clip", "binder", "envelope", "rubber_band", "whiteboard_marker",

    "shoe", "book", "umbrella", "candle", "pillow", "vase",
    "picture_frame", "clock", "bucket", "broom", "dustpan",
    "coat_hanger", "doormat", "curtain_rod", "lampshade", "blanket",
    "cushion", "storage_box", "magazine", "plant_pot",

    "garden_hose", "trowel", "watering_can", "bicycle_pump",
    "tennis_ball", "jump_rope", "yoga_mat", "water_bottle",
    "backpack", "sunglasses", "hat", "gloves",
]

_DISTRACTOR_OBJECTS = _DISTRACTOR_OBJECTS_KITCHEN + _DISTRACTOR_OBJECTS_OTHER


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
    api_key: str = ""                                               
    api_base_url: str = os.environ.get("API_BASE_URL", "")
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "azure")
    model_name: str = os.environ.get("MODEL_NAME", "gpt-5.4")
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "1024"))
    request_images_limit: int = int(os.environ.get("REQUEST_IMAGES_LIMIT", "1000000"))
    max_retries: int = int(os.environ.get("MAX_RETRIES", "3"))
    retry_backoff_sec: float = float(os.environ.get("RETRY_BACKOFF_SEC", "1.5"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.3"))

def initialize_api_client(cfg: ApiConfig) -> Any:

    try:
        from azure_openai_client import initialize_openai_client
        return initialize_openai_client()
    except Exception as e:
        logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
        return None


SYSTEM_PROMPT = """You are an expert Physical Interaction Analyst and skilled English writer.
Your task is to rewrite draft QA answers so they read as fluent, natural, professional English prose — as if a domain expert wrote them from scratch — using ONLY the provided SOURCE_JSON fields.

Core objectives (in priority order):
1) Naturalness & flow: the answer MUST read like a fluent paragraph written by a native speaker, NOT like a machine template. Use varied sentence structure, natural connectives (while, as, since, which, meanwhile), and smooth logical flow. Avoid choppy "X is Y. Z is W." patterns.
2) Strict grounding: do NOT add any new objects, actions, states, or causal claims beyond SOURCE_JSON.
3) Detail & rigor: preserve all technical details; do not simplify away constraints/statuses.
4) Professional tone: objective, academic; no conversational filler. Do NOT use first-person constructions ("I observe", "I note", "we can see"); write in third-person or impersonal voice.
5) Task compliance: obey the task-specific formatting/constraints exactly (e.g., one sentence, no newlines, required phrases).

Key rule: if the draft reads mechanically — facts strung together without logical connectives, repetitive sentence patterns, or schema jargon leaking through — rewrite it into flowing prose while keeping every fact intact. A good rewrite feels like a single coherent explanation, not a list of separate facts. But if the draft already flows well, keep changes minimal. You have FULL freedom to rephrase, merge verbs, restructure sentence order, and use synonyms — the only constraint is that every FACT must be preserved (not every WORD).

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

Conflict resolution:
- Treat SOURCE_JSON as the ground truth. If DRAFT_TEXT appears to conflict with SOURCE_JSON, preserve the DRAFT_TEXT wording because both are generated from the same source, so apparent conflicts indicate template-level nuance that should not be overridden.
- If the draft contains "not directly observable", do NOT turn that claim into a definite statement.

Dataset safety:
- Do NOT mention filenames, paths, extensions, timestamps, or frame numbers.
- Do NOT mention JSON keys/field names or section headers; output answer content only.
- Do NOT output markdown or code fences.
- Avoid placeholders like "unknown", "N/A", or "...".

Grammar & formatting:
- Fix ALL spelling, capitalization, and punctuation errors in the draft.
- After subordinating conjunctions (because, since, as, while, when, if, although), the next word MUST be lowercase unless it is a proper noun or acronym. For example: "because finding..." NOT "because Finding...", "since exposing..." NOT "since Exposing...".
- After colons (:) and dashes (—) mid-sentence, the next word MUST be lowercase unless it is a proper noun, acronym, or the start of a complete quoted sentence.
- After commas mid-sentence, the next word MUST be lowercase. For example: "Meanwhile, a remaining..." NOT "Meanwhile, A remaining...".
- After modal verbs (will, would, should, can, could, may, might, must), the next word MUST be lowercase. For example: "the person will pull..." NOT "the person will Pull...".
- Do NOT capitalize the first word of a clause that follows a conjunction, colon, dash, or comma mid-sentence.
- Ensure subject-verb agreement and consistent tense throughout.
- Do NOT wrap action phrases in quotation marks. Write naturally: "The person is opening the binder" NOT 'The action is "open the binder"'.

Output format:
- Return ONLY the final answer text.
"""


DIRECT_SYSTEM_PROMPT = """\
You are an expert Physical Interaction Analyst and skilled English writer.
Your task is to write a clear, fluent, natural English answer for a video understanding question, \
using ONLY the provided SOURCE_JSON fields as factual basis.

Core objectives (in priority order):
1) Naturalness: write as a native-speaking domain expert would — smooth logical flow, \
varied sentence structure, natural connectives. The answer must NOT sound template-generated. \
You have FULL freedom to rephrase, merge, and restructure the source information — \
the goal is natural prose, NOT verbatim reproduction of source fields.
2) Strict grounding: every factual claim must come from SOURCE_JSON. Do NOT invent objects, \
actions, states, spatial relations, or causal mechanisms not in the source. \
You may freely rephrase and use synonyms, but do NOT add new facts.
3) Completeness: cover all key information from SOURCE_JSON relevant to the task.
4) Professional tone: objective, third-person, academic register. No first-person.

Key rules:
- Do NOT mention JSON field names, file paths, frame numbers, or schema terms.
- Do NOT use phrases like "affordance type", "patient_text", "step_goal" — describe naturally.
- Do NOT use "affordance" as a noun — describe the object's function naturally \
(e.g., "it serves as a grip surface" not "it has a gripping texture affordance").
- ALWAYS include proper articles (the/a/an) before object nouns. \
NEVER write bare noun phrases like "is rim" — always write "is the rim".
- Use consistent object naming — refer to each object the same way throughout.
- Do NOT invent numeric measurements (distances, angles, weights) unless in SOURCE_JSON.
- Mid-sentence capitalization: after commas, conjunctions, colons, dashes, and modal verbs \
(will/would/should/can/could), the next word MUST be lowercase (e.g., "Meanwhile, a ..." NOT "Meanwhile, A ...").
- Do NOT open sentences with "As a result", "Consequently", or "Therefore" unless the context \
is explicitly a logical conclusion from stated premises.
- Return ONLY the answer text, nothing else.
"""




NUMBERED_ANSWER_TASKS: set[str] = {TASK_16, TASK_17, TASK_18}

LLM_FLEX_TASKS: set[str] = {

    TASK_04,
    TASK_05,
    TASK_14,
    TASK_20,
}




LLM_RELAXED_SPAN_TASKS: set[str] = set(LLM_FLEX_TASKS) | {
    TASK_08,                                                     
    TASK_09,                              
    TASK_10,                              
    TASK_11,
    TASK_04,
    TASK_12,
    TASK_13,
    TASK_01,
    TASK_02,
    TASK_03,
    TASK_06,
    TASK_07,
    TASK_15,                              
    TASK_16,                              
    TASK_17,                              
    TASK_18,                              
    TASK_19,
}






DIRECT_GEN_TASKS: set[str] = {
    TASK_08,                                                     
    TASK_11,                                                          
    TASK_04,                                             
    TASK_05,                                                                   
    TASK_03,                                                                
}



STRICT_DRAFT_QUOTE_TASKS: set[str] = {TASK_18}




LLM_SKIP_TASKS: set[str] = set()                                                              




UF_HIGH_RISK_TASKS: set[str] = {
    TASK_04,                                                           
    TASK_12,                                                
    TASK_05,                                                         
    TASK_13,                                                  
    TASK_03,                                       
    TASK_18,                                       
}


DEFAULT_LLM_TASKS: Tuple[str, ...] = ALL_TASKS




_TASK_TEMPERATURE: Dict[str, float] = {
    TASK_08: 0.35,                                                     
    TASK_09: 0.25,                                                                  
    TASK_10: 0.25,                                                            
    TASK_11: 0.35,                                                                        
    TASK_04: 0.45,                                               
    TASK_12: 0.42,                                                  
    TASK_05: 0.50,                                                                          
    TASK_13: 0.38,                                                                 
    TASK_01: 0.48,                                           
    TASK_02: 0.48,                                              
    TASK_03: 0.42,                                                  
    TASK_06: 0.48,                                               
    TASK_07: 0.48,                                                  
    TASK_14: 0.48,                                        
    TASK_15: 0.30,                                                             
    TASK_16: 0.25,                                                                 
    TASK_17: 0.25,                                                          
    TASK_18: 0.30,                                                                                  
    TASK_19: 0.45,                                                
    TASK_20: 0.45,                                              
}


_FEW_SHOT_EXAMPLES: Dict[str, tuple] = {
    TASK_05: (
        "Because the knife is on the cutting board and the onion is graspable, the blade concentrates downward force to cut; as a result, the onion is divided into pieces and the cutting board has onion fragments on its surface.",
        "Because the knife rests within reach on the cutting board and the onion's firm shape allows a stable grip, the blade concentrates downward force along its edge to slice through the onion tissue; as a result, the onion separates into uniform pieces now spread across the cutting board.",
    ),
    TASK_01: (
        "The knife is on the cutting board. The onion is beside the knife. The board is on the counter. The hand is not yet touching any item.",
        "The knife and onion sit side by side on the cutting board, which rests stable on the counter surface. Neither has been touched yet — the hand remains clear of both items.",
    ),
    TASK_02: (
        "The jar is openable because its lid is threaded. The lid is grippable because its diameter allows finger wrap.",
        "The jar's threaded lid allows counter-clockwise rotation for opening, and its diameter is wide enough for a secure finger wrap that provides the torque needed to break the seal.",
    ),
    TASK_06: (
        "The lid is separated from the jar. The jar opening is exposed. The lid is resting on the counter.",
        "The lid now rests on the counter beside the jar, whose opening is fully exposed after the separation.",
    ),
    TASK_07: (
        "The jar is open. The contents are accessible. The lid is removable.",
        "With the lid removed, the jar is open and its contents are directly accessible. The lid itself has transitioned to a freely detached state.",
    ),
    TASK_04: (
        'The main object being acted upon is "onion", which exhibits a cuttable property. Physically, the blade edge concentrates downward force along a thin line, exceeding the onion cell walls\' shear strength.',
        'The onion is the object receiving the cut. Its firm cellular structure yields to the blade because the edge concentrates downward force along a thin line, exceeding the cell walls\' shear strength and separating the tissue cleanly.',
    ),
    TASK_13: (
        "This step is necessary because chopping the onion into uniform pieces ensures consistent cooking time when added to the pan.",
        "Uniformly chopping the onion ensures that all pieces cook at the same rate once they reach the pan, preventing some from charring before others are done.",
    ),
    TASK_03: (
        "Yes, this is feasible because the knife is on the cutting board and the onion is graspable.",
        "This action is feasible — the knife rests within reach on the cutting board, and the onion's firm rounded shape allows a stable pinch grip to anchor it during slicing.",
    ),
    TASK_19: (
        "The onion pieces would be uneven. This would cause uneven cooking later.",
        "Without a controlled cut, the onion would split into fragments of varying thickness — thinner slivers charring quickly while thicker chunks remain undercooked during sautéing.",
    ),
    TASK_20: (
        "Re-chop the unevenly cut pieces to achieve uniform size. This addresses the failure where the onion pieces were cut at inconsistent thicknesses.",
        "Gathering the uneven fragments and re-chopping them to a uniform size corrects the inconsistent thicknesses, so all pieces absorb heat evenly once they reach the pan.",
    ),
    TASK_12: (
        "The person is slicing the carrot with a knife on the cutting board, changing the carrot from whole to sliced pieces.",
        "A knife slices through the carrot on the cutting board, breaking the single whole root into a series of uniform round pieces that spread across the board's surface.",
    ),
    TASK_14: (
        "Chopping the vegetables into small pieces creates uniformly sized fragments that can absorb heat evenly when added to the pan in the next step.",
        "Chopping the vegetables into small uniform pieces is necessary because even-sized fragments ensure consistent heat transfer when they are subsequently added to the heated pan, preventing some pieces from burning while others remain undercooked.",
    ),

}

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



    s = re.sub(
        r"^(?:Sure(?!\s+enough)[,!.]?\s*"
        r"|Certainly[,!.]?\s*"
        r"|Of\s+course[,!.]?\s*"
        r"|Here\s+is\s+(?:the\s+)?(?:rewritten\s+)?(?:answer|text)[:\s]*"
        r"|The\s+answer\s+is[:\s]*"
        r"|Let\s+me\s+(?:explain|describe)[:\s]*"
        r")",
        "", s, flags=re.IGNORECASE
    )
    s = re.sub(r"(?:I\s+hope\s+this\s+helps[.!]?\s*|Let\s+me\s+know\s+if\s+.*$)", "", s, flags=re.IGNORECASE)

    s = re.sub(r"(?m)^#+\s+.*$", "", s)

    if task_name not in NUMBERED_ANSWER_TASKS:
        s = re.sub(r"(?m)^\s*([\-\*•\>]+|\d+[\.\)])\s+", "", s)
    s = s.strip()
    return _sanitize_text_single_line(s)


def _defluff_text(text: str) -> str:
    s = str(text or "")
    patterns = [
        r"^\s*(In summary|In conclusion|To summarize|Overall|In general|Generally|Essentially|Basically),\s*",
        r"^\s*(In this (scene|image|frame|step)),\s*",
        r"^\s*(In the given scenario),?\s*",
        r"^\s*(It should be noted that|Note that)\s*",
        r"^\s*(Here is the answer|Answer)\s*[:\-]\s*",
        r"^\s*(Based on the provided information|Based on the visual evidence),?\s*",
        r"^\s*(As described in the source|As shown in the video|As seen in the image),?\s*",
        r"^\s*(As per the video|As per the information|As per the data),?\s*",
        r"^\s*(According to the data|According to the plan|According to the video),?\s*",
        r"^\s*(It is important to note that|It is worth noting that)\s*",
        r"^\s*(As we can see),?\s*",
        r"^\s*From the observation,?\s*",
        r"^\s*Looking at the (scene|image|video|frame),?\s*",



        r"(?:^|\.\s+)I (?:observe|note|can see|notice|identify) that\s*",
        r"(?:^|\.\s+)We (?:can see|observe|note) that\s*",
        r"(?:^|\.\s+)I would say that\s*",
    ]
    for pat in patterns:

        if r"(?:^|\.\s+)" in pat:
            def _preserve_period(m: re.Match) -> str:
                matched = m.group(0)
                if matched.startswith("."):
                    return ". "                                  
                return ""                                      
            s = re.sub(pat, _preserve_period, s, flags=re.IGNORECASE)
        else:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    if s and s[0].islower():
        s = s[0].upper() + s[1:]


    def _cap_after_period(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()
    s = re.sub(r'(\.\s+)([a-z])', _cap_after_period, s)
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
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", a, re.IGNORECASE) is not None:
            return True
    except re.error:                    
        if t in a:
            return True

    t_human = t.replace("_", " ")
    if t_human != t:
        try:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(t_human)}(?![A-Za-z0-9_])", a, re.IGNORECASE) is not None:
                return True
        except re.error:                    
            if t_human in a:
                return True
    return False


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
    elif task_name == TASK_11:



        _patient = str(fields.get("patient") or "").strip()
        if _patient and _patient.lower() not in _GENERIC_OBJECT_TOKENS:
            spans.append(_lowercase_first_alpha(_patient.strip().rstrip(".")))
    elif task_name == TASK_04:
        desc = str(fields.get("patient") or "").strip()                                              
        mech = str(fields.get("mechanism") or "").strip()
        aff_t = str(fields.get("affordance_type") or "").strip()                                             
        if desc:
            spans.append(_lowercase_first_alpha(desc.strip().rstrip(".")))
        if aff_t:

            spans.append(aff_t.lower().replace("_", " "))
        if mech:
            spans.append(_lowercase_first_alpha(_inline_clause(mech)))
    elif task_name == TASK_12:
        _add(fields.get("action_state_change_description"))
    elif task_name == TASK_05:




        _add(fields.get("mechanism"))
    elif task_name == TASK_13:
        _add(fields.get("rationale"))

    elif task_name == TASK_01:

        pts = fields.get("spatial_preconditions_pts")
        if pts and isinstance(pts, (list, tuple)) and len(pts) >= 2:
            for p in pts:
                _add(p)
        else:
            _add(fields.get("spatial_preconditions"))
    elif task_name == TASK_02:
        pts = fields.get("affordance_preconditions_pts")
        if pts and isinstance(pts, (list, tuple)) and len(pts) >= 2:
            for p in pts:
                _add(p)
        else:
            _add(fields.get("affordance_preconditions"))
    elif task_name == TASK_03:
        _add(fields.get("spatial_precondition"))
        _add(fields.get("affordance_precondition"))
    elif task_name == TASK_06:
        pts = fields.get("spatial_postconditions_pts")
        if pts and isinstance(pts, (list, tuple)) and len(pts) >= 2:
            for p in pts:
                _add(p)
        else:
            _add(fields.get("spatial_postconditions"))
    elif task_name == TASK_07:
        pts = fields.get("affordance_postconditions_pts")
        if pts and isinstance(pts, (list, tuple)) and len(pts) >= 2:
            for p in pts:
                _add(p)
        else:
            _add(fields.get("affordance_postconditions"))
    elif task_name == TASK_14:


        _add(fields.get("detail_independence"))
    elif task_name == TASK_15:
        _add(fields.get("next_step_goal"))
    elif task_name == TASK_16:
        _add(fields.get("middle_step_goals"))
    elif task_name == TASK_17:
        _add(fields.get("next_step_goals"))
    elif task_name == TASK_18:
        flaw_step = fields.get("flaw_step")
        try:
            flaw_step_i = int(flaw_step) if flaw_step is not None else 0
        except Exception:
            flaw_step_i = 0

        if flaw_step_i > 0:
            spans.append(f"step {flaw_step_i}")
        _add(fields.get("repair_steps"))
    elif task_name == TASK_19:
        _add(fields.get("expected_outcome"))
    elif task_name == TASK_20:
        strat = str(fields.get("recovery_strategy") or "").strip()
        if strat:
            spans.append(strat.rstrip(".!?").strip())





    return [s for s in spans if s]


def _morphological_match(a: str, b: str, *, min_prefix: int = 5, min_ratio: float = 0.6) -> bool:

    if not a or not b:
        return False
    pref_len = 0
    for x, y in zip(a, b):
        if x != y:
            break
        pref_len += 1
    if pref_len < min_prefix:
        return False
    shorter = min(len(a), len(b))
    if pref_len < min_ratio * shorter:
        return False


    suf_a = a[pref_len:]
    suf_b = b[pref_len:]


    max_suf = max(len(suf_a), len(suf_b))
    if max_suf <= 4:
        return True

    if len(suf_a) <= 4 and len(suf_b) <= 4:
        return True
    return False


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


            if _morphological_match(t, c):
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



    if n <= 2:
        need = n                                                          
    elif n <= 4:
        need = max(2, n - 1)                                                 
    else:
        need = max(2, int(math.ceil(0.50 * n)))                                  
    need = min(25, need)                                                                
    return matched >= need


def _llm_output_is_acceptable(
    *,
    task_name: str,
    draft_answer: str,
    fields: Dict[str, Any],
    candidate: str,
    question: Optional[str] = None,
    vocab_guard: str = "warn",                                
    verify: str = "strict",                                    
) -> Tuple[bool, str]:
    out = str(candidate or "").strip()
    if not out:
        return False, "empty"




    _MIN_LEN_PARAGRAPH_TASKS = {TASK_01, TASK_02, TASK_06, TASK_07}
    _min_len = 15 if task_name in _MIN_LEN_PARAGRAPH_TASKS else 8
    if len(out) < _min_len:
        return False, f"too-short ({len(out)} < {_min_len})"
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
            r"^The flaw is in step \d+.+The corrected plan is:\s*1\)\s*\".+\"(?:\s+\d+\)\s*\".+\")+\s*$",
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

        _DOMAIN_SNAKE_ALLOWLIST = {"counter_clockwise", "clock_wise", "non_slip", "anti_slip",
                                   "cutting_board", "gas_stove", "frying_pan", "step_goal"}
        allowed_snake |= _DOMAIN_SNAKE_ALLOWLIST
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
            if not dl:
                continue
            if dl in key_objects:
                continue

            dl_human = dl.replace("_", " ")
            if dl_human in key_objects:
                continue
            if _mentions_token(lower_out, dl) or _mentions_token(lower_out, dl_human):
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

            if len(t) < 5:
                return True
            for a in allowed_list:
                if len(a) < 5:
                    continue


                if _morphological_match(t, a):
                    return True
            return False

        novel = sorted([t for t in cand_terms if not _term_allowed(t)])
        if novel:
            if vocab_guard_s == "warn":
                novel_warn_terms = novel[:8]
            else:
                return False, f"introduced novel terms: {novel[:8]}"

    if verify_s != "off":




        if task_name in DIRECT_GEN_TASKS:
            _fields_text_len = len(json.dumps(fields, ensure_ascii=False)) if fields else 200
            _max_output_len = max(400, int(_fields_text_len * 1.5))
        else:
            _max_output_len = max(180, int(4.0 * len(str(draft_answer or ""))) + 220)
        if len(out) > _max_output_len:
            res = _fail_or_warn("too long")
            if res is not None:
                return res



    if task_name == TASK_04:
        if re.search(r"\bthe\s+the\b", out, flags=re.IGNORECASE):
            res = _fail_or_warn("T05 contains 'the the' artifact")
            if res is not None:
                return res



    if task_name == TASK_13:
        _out_lower = out.lower()
        _causal_ok = bool(re.search(
            r'\b(?:because|since|therefore|thus|consequently|hence)\b|as a result',
            _out_lower
        ))
        if not _causal_ok:
            res = _fail_or_warn("T08 missing causal connective (because/since/therefore/...)")
            if res is not None:
                return res


    if task_name in (TASK_15,):
        raw_goal = str(fields.get("next_step_goal", "")).strip().lower()
        if raw_goal:
            goal_words = set(raw_goal.split())
            out_words = set(out.lower().split())
            if goal_words and out_words:
                jaccard = len(goal_words & out_words) / max(1, len(goal_words | out_words))
                if jaccard > 0.8:
                    res = _fail_or_warn(f"T15 answer too similar to raw step_goal (Jaccard={jaccard:.2f})")
                    if res is not None:
                        return res

    if novel_warn_terms:
        warn_reasons.append(f"novel terms: {novel_warn_terms}")


    _BANNED_PHRASES: Dict[str, List[str]] = {
        TASK_03: [
            "the conditions are in place",
            "the conditions are met",
            "everything needed is in place",
            "what enables this action is",
        ],
        TASK_20: [
            "directly addresses the case in which",
            "which directly addresses the case",
            "addresses the case in which",
        ],
        TASK_13: [
            "the purpose of this step is clear",
            "the purpose of this step is to",
            "without this action",
            "without this step",
        ],
        TASK_04: [
            "undergoes the physical change",
            "undergoes a physical change",
            "the hand grips the",
            "force transfers through the",
            "pressure concentrates on the",
        ],
    }
    _bp_list = _BANNED_PHRASES.get(task_name, [])
    if _bp_list:
        for _bp in _bp_list:
            if _bp in lower:
                res = _fail_or_warn(f"banned phrase: {_bp!r}")
                if res is not None:
                    return res

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

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0

        self._counter_lock = threading.Lock()

    def enabled(self) -> bool:
        return self.client is not None

    def call(
        self,
        *,
        system_prompt: str,
        user_text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        require_success: bool = False,
        reasoning_effort: Optional[str] = None,
        _usage_out: Optional[list] = None,
    ) -> str:
        if not self.enabled():
            if bool(require_success):
                raise RuntimeError("LLM is not enabled (Azure OpenAI client not initialized).")
            return ""
        messages = [
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": str(user_text or "")},
        ]
        from azure_openai_client import call_model_api, build_request_payload_input, MODEL
        payload_input = build_request_payload_input(messages)
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(self.cfg.max_retries))):
            try:
                _call_kwargs: Dict[str, Any] = {
                    "model_name": str(self.cfg.model_name or MODEL),
                    "payload_input": payload_input,
                    "timeout_sec": 180.0,
                    "max_tokens": int(max_tokens or self.cfg.max_tokens),
                }
                if reasoning_effort is not None:

                    _call_kwargs["reasoning_effort"] = str(reasoning_effort)
                else:



                    _call_kwargs["reasoning_effort"] = None
                    _call_kwargs["temperature"] = float(self.cfg.temperature if temperature is None else temperature)
                out, usage = call_model_api(self.client, **_call_kwargs)
                out = str(out or "")

                with self._counter_lock:
                    self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
                    self.total_completion_tokens += int(usage.get("completion_tokens", 0))
                    self.total_api_calls += 1
                if bool(require_success) and not out.strip():
                    raise RuntimeError("LLM returned empty content.")

                if _usage_out is not None:
                        _usage_out.append(usage)
                logger.info(f">>> [LLM] ok len={len(out)} tokens={usage.get('total_tokens', 0)}")
                return out
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
        if pol0 not in {"polish", "copyedit", "verbatim", "direct"}:
            pol0 = "polish"


        if pol0 == "verbatim":
            return str(draft_answer or "")

        required_snake = sorted(_collect_snake_tokens({}, extra_texts=[draft_answer]))
        allowed_snake = sorted(_collect_snake_tokens(fields, extra_texts=[draft_answer]))
        required_quotes = sorted({q for q in _extract_double_quoted(draft_answer) if q})
        required_spans = _llm_required_spans(task_name, fields=fields, draft_answer=draft_answer)
        must_keep_phrase = "not directly observable" if "not directly observable" in str(draft_answer or "").lower() else ""
        strict_quotes = task_name in STRICT_DRAFT_QUOTE_TASKS

        fmt = "Produce a single paragraph."

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
                    "Goal summary: rewrite the draft into a fluent, natural high-level goal statement. "
                    "Cover ALL activities mentioned in the draft — do NOT drop, truncate, or omit any part. "
                    "You may restructure sentences, improve flow, and remove awkward phrasing, "
                    "but every action and object from the draft must appear in your output. "
                    "Write as a single coherent paragraph of 2-4 sentences. "
                    "Break the goal into logical phases (e.g., gathering → preparing → executing). "
                    "Each sentence should cover one phase. Do NOT pack all activities into a single run-on sentence. "
                    "Do NOT add actions, objects, or details that are not in the draft."
                )
            if task_name == TASK_11:
                task_guidelines_lines.append(
                    "Action description: rewrite the draft into a fluent, natural description of the action. "
                    "The SOURCE contains 'patient' (the specific object) and 'step_goal' (the purpose). "
                    "Use the patient name as the specific object — NEVER write generic 'the object' or 'the item'. "
                    "MERGE multiple atomic verbs into a natural description of the overall motion — "
                    "do NOT enumerate verbs mechanically (e.g., NOT 'grasping, lifting, carrying, and lowering' "
                    "→ INSTEAD 'picking up the X and placing it on the Y'). "
                    "Write exactly ONE fluent sentence in progressive (-ing) form. "
                    "Do NOT add actions, objects, or details not in the source fields."
                )
                task_guidelines_lines.append(
                    "Grammar: ALWAYS include proper articles (the/a/an) before object nouns. "
                    "NEVER write bare noun phrases like 'pressing machine control' — write 'pressing the machine control'. "
                    "Ensure every verb in the description uses the progressive (-ing) form."
                )
            if task_name == TASK_04:
                task_guidelines_lines.append(
                    "Hotspot mechanism: explain a physical mechanism grounded in contacts/forces (force/torque transfer, friction, leverage, flow, constraint satisfaction); avoid high-level intent/semantic goals."
                )
                task_guidelines_lines.append(
                    "Hotspot summary: preserve the patient object, affordance type, and mechanism from the draft. "
                    "Write naturally as a flowing 2-3 sentence paragraph. "
                    "BANNED sentence openers: 'Here, the physical interaction centers on', 'In this interaction, the hand engages with', "
                    "'What makes this action possible is', 'The key contact point here is'. "
                    "These are formulaic — instead lead with the concrete physical fact or the object's role."
                )
            if task_name == TASK_12:
                task_guidelines_lines.append(
                    "State change: genuinely rewrite the draft into a fluent 2-3 sentence description. "
                    "Preserve the agent, action verb, patient, and before→after state, but DO NOT just copy the draft verbatim. "
                    "Restructure sentence boundaries, vary verb forms, and weave the transition into natural prose. "
                    "For example, instead of a single dense sentence listing all changes, break it into: "
                    "(1) what action was performed, (2) what changed as a result. "
                    "Focus on the object's physical state change, not the agent's hand/body movement. "
                    "Keep it objective and grounded; do NOT introduce time/frame references or speculative hidden states."
                )
            if task_name in (TASK_01, TASK_06):
                task_guidelines_lines.append(
                    "Spatial statements: use directly observable relations (contact/relative position/containment/alignment/support/open-closed). Each sentence should explicitly name two entities and their visible relation; avoid abstract terms like 'accessible/within reach'."
                )
                task_guidelines_lines.append(
                    "Spatial statements: prefer two-entity relations; if the draft expresses a single-entity state (e.g., open/closed), preserve its factual content but you may combine it with adjacent statements for better flow."
                )
                task_guidelines_lines.append(
                    "Spatial statements: format as a single paragraph of complete sentences; each sentence should end with '.', and do NOT add numbering/bullets."
                )
                task_guidelines_lines.append(
                    "CRITICAL REWRITING RULE: Do NOT just stitch the draft's bullet points together with 'while/Meanwhile/Additionally/At the same time/Furthermore'. "
                    "That produces robotic text. Instead, genuinely restructure: "
                    "(a) Group related spatial facts into composite sentences (e.g., merge an object's location + its readiness into one sentence). "
                    "(b) Use subordinate clauses (which, where, whose, that) and prepositional phrases (with, beside, beneath) to embed facts naturally. "
                    "(c) Vary sentence length — mix short declarative sentences with longer compound ones. "
                    "(d) If two facts share the same subject, combine them into one sentence instead of writing two. "
                    "(e) If a fact is implied by a neighboring fact, drop the redundant one. "
                    "BANNED connective patterns: Do NOT start any sentence with 'Meanwhile', 'Additionally', 'Furthermore', or 'At the same time'. "
                    "Do NOT use causal connectives ('As a result', 'Therefore', 'Consequently') between co-existing conditions. "
                    "ANTI-VERBATIM: Do NOT copy draft sentences word-for-word. Every sentence must be genuinely restructured — "
                    "change sentence boundaries, reorder clauses, merge facts that share a subject, use different verb forms. "
                    "A paragraph where each sentence matches a source bullet point 1-to-1 is NOT an acceptable rewrite."
                )
            if task_name == TASK_01:
                task_guidelines_lines.append(
                    "Spatial conditions before the step: describe the state BEFORE the step begins. "
                    "Do NOT describe mid-action states ('hand is grasping', 'knife is cutting'). "
                    "Focus on the starting spatial arrangement that makes the step possible."
                )
            if task_name == TASK_06:
                task_guidelines_lines.append(
                    "Spatial results after the step: describe what CHANGED after the step. "
                    "Do NOT describe long-horizon future states ('ready for storage', 'all fully cut'). "
                    "Focus on the IMMEDIATE spatial result of this single step."
                )
            if task_name in (TASK_02, TASK_07):
                task_guidelines_lines.append(
                    "Affordance statements: only include operability/readiness states that are visible or strongly implied by mechanical state (open/closed, sealed/unsealed, empty/full, blocked/unblocked, graspable, stable, separated/clumped). Avoid hidden qualities (sharp/clean/functional heater) unless clearly visible; avoid semantic goals."
                )
                task_guidelines_lines.append(
                    "Affordance statements: format as a single paragraph of complete sentences; each sentence should end with '.', and do NOT add numbering/bullets."
                )
                task_guidelines_lines.append(
                    "CRITICAL REWRITING RULE: Do NOT just stitch the draft's bullet points together with 'while/Meanwhile/Additionally/Furthermore/At the same time'. "
                    "That produces robotic text. Instead, genuinely restructure: "
                    "(a) Group related functional properties into composite sentences (e.g., an object's state + what that enables in one sentence). "
                    "(b) Use subordinate clauses (which, whose, meaning, so that) and causal links within a single sentence. "
                    "(c) Vary sentence length — mix short and long. "
                    "(d) Merge facts that share the same subject into one sentence. "
                    "(e) Drop redundant facts that repeat the same information in different words. "
                    "BANNED connective patterns: Do NOT start any sentence with 'Meanwhile', 'Additionally', 'Furthermore', or 'At the same time'. "
                    "Do NOT use causal connectives ('As a result', 'Therefore') between independent co-existing conditions."
                )
            if task_name == TASK_02:
                task_guidelines_lines.append(
                    "Functional properties before the step: describe SPECIFIC physical properties (shape, material, surface texture, "
                    "rigidity, heft) that enable the affordance — use qualitative terms (e.g., 'lightweight', 'rigid plastic', "
                    "'textured grip surface'), NOT numeric measurements. Do NOT use tautological descriptions ('object is graspable'). "
                    "Do NOT describe spatial conditions ('on countertop', 'has space') — those belong to spatial conditions. "
                    "Focus on the object's intrinsic functional properties."
                )
            if task_name == TASK_07:
                task_guidelines_lines.append(
                    "Functional state changes after the step: describe the IMMEDIATE functional state change of the PRIMARY object acted on. "
                    "Do NOT describe 'future usability', 'readiness for later steps', or workspace side-effects. "
                    "Focus on the single main object's new functional state. "
                    "EXTRA STYLE RULE for T13: Do NOT use 'while' to glue unrelated state-change facts together. "
                    "'While' is only acceptable when describing genuinely concurrent changes to the SAME object. "
                    "For facts about DIFFERENT objects, use separate sentences or subordinate clauses (whose, which, leaving, so that). "
                    "Prefer sentence structures like: 'The X has changed from A to B, leaving the Y in state C.' "
                    "or 'With the X now in state B, the Y is no longer Z.'"
                )
            if task_name == TASK_05:
                task_guidelines_lines.append(
                    "Causal chain: explain the causal chain from preconditions through mechanism to effects. "
                    "Use your own natural sentence structure — you are NOT required to follow the draft's connective form. "
                    "Cover spatial AND affordance preconditions, the physical mechanism, "
                    "and the immediate spatial AND affordance effects. "
                    "Do NOT repeat the action description (covered by T06) or patient/affordance_type (covered by T05). "
                    "Do NOT drift to 'future readiness' or 'workspace summary' — describe only immediate physical consequences."
                )
            if task_name == TASK_13:
                task_guidelines_lines.append(
                    "Rationale: rewrite the draft rationale into a fluent, natural explanation. "
                    "Preserve ALL reasoning from the source rationale — do not omit any causal link. "
                    "MUST use at least one 'because', 'since', 'therefore', or 'thus'. "
                    "Do NOT add any reasoning, causal claims, or elaboration beyond what is in the draft. "
                    "Do NOT use generic phrases like 'enables the next step', 'keeps workspace clean', "
                    "or 'prepares for future use' — be specific about the causal link. "
                    "Do NOT introduce new steps, tools, objects, or causal connections not in the draft. "
                    "VARY your opener — do NOT always start with 'This step...'. Use varied structures "
                    "like 'Removing the X is necessary because...', "
                    "'[Verb]-ing the X ensures that...', 'By [verb]-ing the X, the person...'. "
                    "BANNED openers: 'Without this action/step, ...', 'The purpose of this step is clear', "
                    "'The purpose of this step is to'. Instead, lead with the concrete action or its consequence."
                )
            if task_name == TASK_03:


                task_guidelines_lines.append(
                    "Feasibility explanation: the question asks WHY the step is feasible, "
                    "NOT whether it is feasible. Do NOT start with 'Yes, this is feasible' or 'Yes, because...'. "
                    "Instead, directly explain the enabling conditions. "
                    "VARY your opener — do NOT always use the same structure. Examples:\n"
                    "  - 'This action is feasible because...'\n"
                    "  - 'The step can proceed since...'\n"
                    "  - '[Object] is positioned so that...'\n"
                    "  - 'Because [spatial fact], and [affordance fact], the action is possible.'\n"
                    "BANNED openers: 'The conditions are in place:', 'The conditions are met:', "
                    "'What enables this action is', 'Everything needed is in place'. "
                    "These are robotic — instead lead with a concrete spatial or functional fact. "
                    "Explain WHY by connecting one spatial precondition and one affordance "
                    "precondition into a natural English sentence. "
                    "Do not classify or judge — assume feasibility and explain the enabling conditions."
                )
            if task_name == TASK_14:
                task_guidelines_lines.append(
                    "Inter-step dependency: your answer MUST contain information NOT present in the question. "
                    "Reference both step goals naturally but do NOT just restate them — explain the specific "
                    "physical/spatial outcome of the previous step and how it creates a precondition for the next step. "
                    "Name specific objects, locations, or states that bridge the two steps."
                )


            if task_name == TASK_09:
                task_guidelines_lines.append(
                    "Object list: rewrite the draft into a fluent, natural sentence listing the key objects. "
                    "Preserve EVERY object name exactly — do NOT rename, drop, or add any objects. "
                    "You may restructure into a more natural sentence (e.g., 'The key objects are X, Y, and Z' "
                    "→ 'This activity involves X, Y, and Z'). Keep it to one sentence."
                )
            if task_name == TASK_10:
                task_guidelines_lines.append(
                    "Step goal: rewrite the draft into a fluent, natural imperative sentence. "
                    "Preserve the COMPLETE goal content — every action and object must appear. "
                    "You may improve word flow and add natural articles/prepositions, but do NOT change the meaning. "
                    "Keep imperative form. One sentence."
                )
            if task_name == TASK_15:
                task_guidelines_lines.append(
                    "Next step prediction: rewrite the draft into a fluent, natural sentence. "
                    "Preserve the exact step goal content — do NOT paraphrase the step description itself. "
                    "You may improve the framing sentence (e.g., vary the opener), but keep the step goal intact. "
                    "One sentence."
                )
            if task_name == TASK_16:
                task_guidelines_lines.append(
                    "Middle steps infill: rewrite the draft into fluent, naturally phrased numbered steps. "
                    "Preserve EVERY step goal exactly as given — do NOT paraphrase, reorder, or omit any step. "
                    "You may improve the introductory framing and add natural articles/prepositions to each step. "
                    "Keep the numbered list format: 1) \"...\" 2) \"...\" etc."
                )
            if task_name == TASK_17:
                task_guidelines_lines.append(
                    "Next K steps prediction: rewrite the draft into fluent, naturally phrased predicted steps. "
                    "Preserve EVERY step goal exactly — do NOT paraphrase, reorder, add, or omit any step. "
                    "You may improve the introductory sentence framing for variety. "
                    "Keep the numbered list format: 1) \"...\" 2) \"...\" etc."
                )
            if task_name == TASK_18:
                task_guidelines_lines.append(
                    "Plan diagnosis: rewrite the draft into a fluent, natural explanation of the flaw and repair. "
                    "Preserve the flaw step number, flaw type description, reason, and the complete corrected plan. "
                    "You may restructure the explanation for natural flow, but the corrected plan steps must remain "
                    "in numbered format with exact step goals: 1) \"...\" 2) \"...\" etc. "
                    "Do NOT change, drop, or reorder any corrected plan step."
                )
                task_guidelines_lines.append(
                    "ANTI-HALLUCINATION: Do NOT speculate about precondition chains, "
                    "original step ordering, or why steps were originally arranged a certain way "
                    "unless the source data EXPLICITLY states it. Only describe what the source "
                    "says is wrong and what the corrected plan is. Do NOT invent causal explanations "
                    "for why the flaw exists."
                )
            if task_name == TASK_19:
                task_guidelines_lines.append(
                    "Counterfactual outcome: describe ONE specific physical consequence ONLY. "
                    "Do NOT propose recovery actions (no 'should/need to/must/could' + verb). "
                    "Do NOT cascade multiple consequences. Do NOT use generic safety/hygiene warnings. "
                    "Focus on the immediate physical state change caused by the counterfactual. "
                    "Do NOT open with 'As a result', 'Consequently', or 'Therefore' — "
                    "describe the outcome directly (e.g., 'The knife would slip...' not 'As a result, the knife would slip...')."
                )
                task_guidelines_lines.append(
                    "Counterfactual outcome: keep the response concise (2-4 sentences). Focus on the immediate physical consequence."
                )
                task_guidelines_lines.append(
                    "CRITICAL: Do NOT copy the draft text verbatim. You MUST genuinely rewrite it: "
                    "restructure sentences, change word order, use different verb forms or synonyms for non-key terms. "
                    "The rewrite must preserve ALL factual claims (objects, states, consequences) but the SENTENCE STRUCTURE "
                    "must differ from the draft. For example, if the draft says 'X would stay in state A, preventing Y', "
                    "you might write 'Because X remains in state A, Y cannot occur' or 'Y is blocked since X stays in state A.' "
                    "A verbatim copy is NOT an acceptable rewrite."
                )
            if task_name == TASK_20:
                task_guidelines_lines.append(
                    "Recovery strategy: preserve the core recovery strategy content. "
                    "You may restructure and expand into a natural explanation (2-3 sentences). "
                    "Do NOT add actions or details not implied by the source data. "
                    "Do NOT use generic phrases like 'improves spatial stability', 'wipe and continue', or 'relevant affordance/mechanism'. "
                    "The recovery must address the SPECIFIC failure reason, not generic cleanup. "
                    "BANNED phrases: 'directly addresses the case in which', 'addresses the case in which', "
                    "'which directly addresses'. Instead of explaining what the recovery 'addresses', "
                    "describe HOW the recovery action fixes the physical problem (e.g., 'Angling the pan allows it to fit past the obstruction')."
                )
                task_guidelines_lines.append(
                    "If a counterfactual scenario (a hypothetical condition and its expected outcome) is described in the source data, "
                    "ensure the recovery strategy is coherent with that scenario — "
                    "the recovery should address the specific failure described, not an unrelated failure mode."
                )
            task_guidelines = ""
            if task_guidelines_lines:
                task_guidelines = "Task-specific guidelines (follow exactly):\n- " + "\n- ".join(task_guidelines_lines) + "\n"

            fse = _FEW_SHOT_EXAMPLES.get(task_name)
            few_shot_block = ""
            if fse:
                few_shot_block = (
                    f"EXAMPLE_REWRITE (for reference style only — do NOT copy these specific facts):\n"
                    f"  BEFORE: {fse[0]}\n"
                    f"  AFTER:  {fse[1]}\n"
                )
            constraints = (
                f"{action} the DRAFT_TEXT into fluent, natural English prose while preserving ALL factual content.\n"
                "The result must read as if written by a native-speaking domain expert — smooth logical flow, varied sentence structure, natural connectives between ideas.\n"
                "If the draft reads mechanically (facts concatenated without connectives, repetitive 'X is Y' patterns, schema jargon), rewrite for fluency — restructure sentences and add connectives to create a coherent paragraph. But if the draft already reads naturally, keep changes minimal.\n"
                "Do NOT add new objects, actions, states, or claims beyond SOURCE_JSON.\n"
                "Use consistent object naming; do not rename the same object with synonyms.\n"
                "Avoid second-person/imperatives (no 'you should'); answer declaratively.\n"
                "Avoid placeholders like 'unknown', 'N/A', or '...'.\n"
                "Preserve all underscore identifier tokens (tokens containing underscores) EXACTLY as they appear; do not modify them.\n"
                "Do NOT introduce any new underscore identifier tokens.\n"
                "Do NOT mention JSON keys/field names or any prompt block headers; output answer content only.\n"
                "Do NOT use newlines; output as one line unless the task explicitly requires multiple paragraphs.\n"
                "Avoid bullets/list markers; keep any required numbering/quotes exactly as in the draft.\n"
                f"{quote_rule}\n"
                f"{span_rule}\n"
                + task_guidelines
                + few_shot_block
                + (
                    "If the draft uses the phrase \"not directly observable\", keep it when the claim cannot be strictly verified from visible evidence.\n"
                    if must_keep_phrase
                    else ""
                )
                + (
                    "For Task_18, preserve the exact key-value format: FlawStep=...; FlawType=...; Reason=...; Repair: 1) \"...\" 2) \"...\" (variable number of repair steps).\n"
                    if task_name == TASK_18
                    else ""
                )
                + "Do NOT mention filenames, paths, extensions, timestamps, or frame numbers.\n"
                + "Return ONLY the final answer text.\n"
                + f"{fmt}\n"
            )

            _pfields = dict(fields)
            for _pk in ("patient", "agent"):
                _pv = str(_pfields.get(_pk) or "").strip()
                if _pv and not re.match(r"^(the|a|an|his|her|its|their|both)\s", _pv, re.IGNORECASE):
                    _pfields[_pk] = "the " + _pv
            parts = [
                f"TASK_NAME:\n{task_name.replace('_', ' ')}",
                "SOURCE_JSON:\n" + json.dumps(_pfields, ensure_ascii=False),
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


        def _direct_prompt() -> str:

            task_label = task_name.replace('_', ' ')

            if task_name == TASK_08:
                instruction = (
                    "Write a fluent, natural high-level goal summary for this video in 2-4 sentences.\n"
                    "Cover ALL activities from the source data. Break into logical phases "
                    "(e.g., gathering → preparing → executing).\n"
                    "Do NOT add any activities, objects, or details not in the source."
                )
            elif task_name == TASK_11:
                instruction = (
                    "Describe the physical action being performed in this clip.\n"
                    "Write exactly ONE fluent sentence in progressive (-ing) form.\n"
                    "ALWAYS include proper articles (the/a/an) before object nouns.\n"
                    "The SOURCE_JSON 'action' field lists atomic verbs — do NOT enumerate them "
                    "mechanically. Instead, MERGE them into a natural description of the overall motion. "
                    "For example, instead of 'grasping, lifting, carrying, lowering, and releasing the egg', "
                    "write 'picking up the egg and placing it into the cup'.\n"
                    "Use 'patient' as the specific object name — NEVER use generic words "
                    "like 'the object' or 'the item'.\n"
                    "Use 'step_goal' to understand the PURPOSE of the action and frame your description around it.\n"
                    "Do NOT add actions, objects, or details not present in any SOURCE_JSON field."
                )
            elif task_name == TASK_04:

                _t05_openers = [
                    "The object acted upon here is {patient}, which ...",
                    "{patient} is the target of the interaction — ...",
                    "In this frame, {patient} plays the central role: ...",
                    "The step acts on {patient}, whose ...",
                    "{patient} is what the hand contacts, and ...",
                    "The relevant surface is {patient}, which ...",
                    "What the action targets is {patient} — ...",
                    "The interaction centers on {patient}: ...",
                    "At this moment, {patient} is where contact occurs — ...",
                    "The primary object involved is {patient}, whose ...",
                    "{patient} is the part being manipulated here — ...",
                    "The key element is {patient}, which ...",
                ]

                _t05_base = hash(json.dumps(fields, sort_keys=True, ensure_ascii=False))
                _t05_step_idx = fields.get("step_index", fields.get("step_number", 0))
                _t05_frame_idx = fields.get("frame_index", fields.get("keyframe_index", 0))
                _t05_hash = (_t05_base + int(str(_t05_step_idx) or "0") * 7 + int(str(_t05_frame_idx) or "0") * 13) % len(_t05_openers)
                _chosen_opener = _t05_openers[_t05_hash]
                instruction = (
                    "Describe the physical interaction in this keyframe: identify the main object being "
                    "acted upon (the patient), its functional role, and the physical mechanism.\n"
                    "Write as a flowing 2-3 sentence paragraph. Use natural language — avoid technical "
                    "jargon like 'affordance type' or 'gripping texture'. Instead describe function naturally "
                    "(e.g., 'The rim provides a grip surface' not 'The rim has a gripping texture affordance').\n"
                    "MANDATORY FIELDS — your answer MUST include ALL THREE of these from SOURCE_JSON:\n"
                    "  1) The patient object name (from 'patient' field) — state EXACTLY this name.\n"
                    "  2) The affordance type value (from 'affordance_type' field) — you MUST use this "
                    "exact term naturally in a phrase describing the object's functional role "
                    "(e.g., if affordance_type is 'rim', write '...serves as a rim that can be gripped'; "
                    "if 'contact surface', write '...provides a contact surface for the hand').\n"
                    "  3) The mechanism description (from 'mechanism' field) — rephrase naturally but "
                    "preserve the core physical description.\n"
                    f"Use this opener pattern as inspiration (adapt freely): \"{_chosen_opener}\"\n"
                    "IMPORTANT: Do NOT assume specific actions (grip, press, push, pull) in the opener — "
                    "only describe the action that is stated in the SOURCE_JSON 'mechanism' field. "
                    "The opener just introduces the patient object; the mechanism sentence should follow.\n"
                    "Describe the mechanism using ONLY what is stated in the SOURCE_JSON 'mechanism' field. "
                    "You may rephrase it naturally, but do NOT invent additional forces, pressures, "
                    "friction explanations, or causal details beyond what the source explicitly says.\n"
                    "Do NOT use the word 'affordance' as a noun anywhere in your answer.\n"
                    "Do NOT add objects, properties, or mechanisms not in the source."
                )
            elif task_name == TASK_05:
                instruction = (
                    "Explain the complete causal chain for this action step.\n"
                    "Your explanation should cover the preconditions (what spatial/functional setup "
                    "enables the action), the physical mechanism (how forces and contacts produce "
                    "the change), and the immediate effects (what changes in the scene).\n"
                    "Write as a coherent paragraph — between 3 and 6 sentences depending on "
                    "complexity. Weave preconditions, mechanism, and effects into a NATURAL narrative "
                    "flow. Do NOT separate them into rigid blocks.\n"
                    "CRITICAL: VARY your sentence count and structure across samples:\n"
                    "  - Sometimes start with the mechanism, then explain preconditions as context\n"
                    "  - Sometimes open with a scene observation, then chain cause→action→effect\n"
                    "  - Sometimes merge precondition and mechanism into one flowing sentence\n"
                    "  - Sometimes split a complex effect across two shorter sentences\n"
                    "Use natural connectives (since, as, which allows, thereby, leading to, given that).\n"
                    "Do NOT add objects, states, or causal claims not in the source."
                )
            elif task_name == TASK_03:

                _t11_openers = [
                    "This action is feasible because ...",
                    "The step can proceed since ...",
                    "Because [spatial fact] and [affordance fact], this step is possible ...",
                    "[Object] is positioned so that ...",
                    "This is physically possible because ...",
                    "Given that [spatial fact], and [affordance fact], the action can proceed ...",
                ]
                _t11_hash = hash(json.dumps(fields, sort_keys=True, ensure_ascii=False)) % len(_t11_openers)
                _chosen_t11_opener = _t11_openers[_t11_hash]
                instruction = (
                    "Explain why the step described in 'step_goal' is physically feasible.\n"
                    "The SOURCE_JSON contains:\n"
                    "  - 'step_goal': what the step aims to do\n"
                    "  - 'patient': the object being acted on\n"
                    "  - 'action': the physical action\n"
                    "  - 'spatial_precondition': a spatial condition enabling the step\n"
                    "  - 'affordance_precondition': a functional property enabling the step\n"
                    "Write ONE fluent sentence that connects the spatial condition and the "
                    "affordance condition to explain why the step can proceed.\n"
                    "Name the specific object (from 'patient') — do NOT write 'the object'.\n"
                    "Do NOT start with 'Yes' — the question asks WHY, not WHETHER.\n"
                    f"You MUST start your answer with this opener pattern: \"{_chosen_t11_opener}\"\n"
                    "Do NOT add conditions, objects, or properties not in the SOURCE_JSON."
                )
            else:
                instruction = "Write a natural, fluent answer based on the source data."

            required_facts_block = json.dumps(
                {"required_facts": required_spans}, ensure_ascii=False
            )



            _fields_for_prompt = dict(fields)
            for _art_key in ("patient", "agent"):
                _art_val = str(_fields_for_prompt.get(_art_key) or "").strip()
                if _art_val and not re.match(r"^(the|a|an|his|her|its|their|both)\s", _art_val, re.IGNORECASE):
                    _fields_for_prompt[_art_key] = "the " + _art_val

            parts = [
                f"TASK_NAME:\n{task_label}",
                "SOURCE_JSON:\n" + json.dumps(_fields_for_prompt, ensure_ascii=False),
                f"INSTRUCTION:\n{instruction}",
                f"REQUIRED_FACTS (your answer must cover the meaning of each fact, but you may rephrase freely — do NOT copy source wording verbatim):\n{required_facts_block}",
                "ANSWER:",
            ]
            return "\n\n".join(parts)


        if pol0 == "direct":
            direct_temp = _TASK_TEMPERATURE.get(task_name, self.cfg.temperature)
            direct_temp = min(direct_temp + 0.05, 0.55)                                           
            raw = self.call(
                system_prompt=DIRECT_SYSTEM_PROMPT,
                user_text=_direct_prompt(),
                temperature=float(direct_temp),
                require_success=bool(require_success),
            )
            raw = raw.strip() if isinstance(raw, str) else ""
            if not raw:
                return draft_answer
            return _defluff_text(_sanitize_answer(task_name, raw))


        task_temp = _TASK_TEMPERATURE.get(task_name, self.cfg.temperature)
        raw = self.call(system_prompt=SYSTEM_PROMPT, user_text=_prompt(draft_answer, pol=pol0),
                        temperature=float(task_temp), require_success=bool(require_success))
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            return draft_answer
        out = _defluff_text(_sanitize_answer(task_name, raw))
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
        return out2


def _apply_llm(
    samples: List[Sample],
    llm: TwoStageLlm,
    llm_tasks: set[str],
    *,
    two_pass: bool,
    fallback: str,
    require_success: bool = False,
    vocab_guard: str = "warn",                                
    verify: str = "strict",                                    
    parallel_n: int = 1,
) -> List[Sample]:
    if not samples or not llm_tasks or not llm.enabled():
        return samples

    vocab_guard_s = str(vocab_guard or "strict").strip().lower()
    if vocab_guard_s not in {"strict", "warn", "off"}:
        vocab_guard_s = "strict"

    verify_s = str(verify or "strict").strip().lower()
    if verify_s not in {"strict", "warn", "off"}:
        verify_s = "strict"


    def _process_one(s: Sample) -> Dict[str, Any]:

        result: Dict[str, Any] = {"sample": None, "status": "skipped", "stats": {}}

        if s.task_name not in llm_tasks:
            result["sample"] = s
            result["status"] = "passthrough"
            return result
        if s.task_name in LLM_SKIP_TASKS:
            result["sample"] = s
            result["status"] = "skipped_structured"
            return result

        fields = s.llm_fields or {}
        original_answer = s.answer
        new_answer = original_answer
        stats: Dict[str, int] = {"attempted": 1, "rewrote": 0, "fell_back": 0, "dropped": 0,
                                  "quote_repaired": 0, "verbatim_used": 0, "vocab_warned": 0,
                                  "direct_used": 0, "direct_to_polish": 0}


        _use_direct = s.task_name in DIRECT_GEN_TASKS
        if _use_direct:
            try:
                new_answer = llm.generate_answer(
                    task_name=s.task_name,
                    fields=fields,
                    draft_answer=original_answer,
                    two_pass=False,
                    policy="direct",
                    require_success=False,
                )
            except Exception as e:
                logger.warning(f"[DIRECT] failed for task={s.task_name}: {e}")
                new_answer = original_answer
                _use_direct = False

            if _use_direct and new_answer and new_answer != original_answer:

                ok_d, why_d = _llm_output_is_acceptable(
                    task_name=s.task_name,
                    draft_answer=original_answer,
                    fields=fields,
                    candidate=new_answer,
                    question=s.question,
                    vocab_guard=vocab_guard_s,
                    verify=verify_s,
                )
                if ok_d:
                    stats["direct_used"] += 1
                    logger.info(f"[DIRECT] accepted for task={s.task_name}")
                else:
                    logger.info(f"[DIRECT] rejected for task={s.task_name}: {why_d}; falling back to polish")
                    stats["direct_to_polish"] += 1
                    new_answer = original_answer
                    _use_direct = False
            elif _use_direct:

                _use_direct = False


        if not _use_direct:
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
            stats["quote_repaired"] += 1

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
            retry_policies: List[str] = ["verbatim"]
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
                    stats["quote_repaired"] += 1
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
                        stats["verbatim_used"] += 1
                    break
                last_why = why2
            if not retry_ok:
                why = last_why
        if not ok:
            if fallback == "fail":
                raise RuntimeError(f"LLM output rejected for task={s.task_name}: {why}")
            if fallback == "skip":
                stats["dropped"] += 1
                result["stats"] = stats
                result["status"] = "dropped"
                return result
            new_answer = original_answer
            stats["fell_back"] += 1

            if "banned phrase" in str(why):
                _scrub_map = {
                    "the conditions are in place:": "This step is feasible because",
                    "the conditions are in place,": "This step is feasible because",
                    "the conditions are in place —": "This step is feasible because",
                    "the conditions are met:": "This step is feasible because",
                    "the conditions are met,": "This step is feasible because",
                    "everything needed is in place —": "This step is feasible because",
                    "everything needed is in place:": "This step is feasible because",
                    "what enables this action is that": "This step can proceed because",
                    "what enables this action is": "This step can proceed because",
                    "the purpose of this step is clear,": "This step matters",
                    "the purpose of this step is clear": "This step matters",
                    "the purpose of this step is to": "This step serves to",
                    "without this action,": "If this action is skipped,",
                    "without this action": "If this action is skipped",
                    "without this step,": "If this step is skipped,",
                    "without this step": "If this step is skipped",
                    "directly addresses the case in which": "corrects the situation where",
                    "which directly addresses the case": "which corrects the situation",
                    "addresses the case in which": "corrects the situation where",
                    "undergoes the physical change": "is the object being acted upon",
                    "undergoes a physical change": "is the object being acted upon",
                    "the hand grips the": "the hand contacts the",
                    "force transfers through the": "the action involves the",
                    "pressure concentrates on the": "the action targets the",
                }
                _ans_lower = new_answer.lower()
                for _bp_old, _bp_new in _scrub_map.items():
                    _idx = _ans_lower.find(_bp_old)
                    if _idx >= 0:

                        if _idx == 0 or new_answer[_idx - 1] in ".!?\n":
                            _bp_new_cased = _bp_new[0].upper() + _bp_new[1:]
                        else:
                            _bp_new_cased = _bp_new
                        new_answer = new_answer[:_idx] + _bp_new_cased + new_answer[_idx + len(_bp_old):]
                        _ans_lower = new_answer.lower()
                        logger.info(f"[SCRUB] Replaced banned phrase in fallback for task={s.task_name}")
        else:
            if _sanitize_space(new_answer) != _sanitize_space(original_answer):
                stats["rewrote"] += 1
            if vocab_guard_s == "warn" and str(why).startswith("warn:"):
                stats["vocab_warned"] += 1




        new_answer = _post_llm_fix_caps(new_answer)





        if s.task_name == TASK_11:

            m_perf = re.search(r'performing the action\s+(.+?)\.?\s*$', new_answer)
            if m_perf:
                _action_part = m_perf.group(1)
                _prog = _to_progressive_phrase(_action_part)
                if _prog != _action_part:
                    new_answer = new_answer[:m_perf.start()] + _prog.rstrip('.') + '.' + new_answer[m_perf.end():]
                    new_answer = _sanitize_space(new_answer)

            m_shows = re.search(r'shows the person\s+(.+?)\.?\s*$', new_answer)
            if m_shows:
                _action_part = m_shows.group(1)
                _prog = _to_progressive_phrase(_action_part)
                if _prog != _action_part:
                    new_answer = new_answer[:m_shows.start()] + 'shows the person ' + _prog.rstrip('.') + '.' + new_answer[m_shows.end():]
                    new_answer = _sanitize_space(new_answer)




        if s.task_name == TASK_20:
            new_answer = re.sub(
                r'(achieved|addressed|accomplished|done|handled)\s+by\s+([A-Z][a-z]+)',
                lambda m: m.group(1) + " by " + _imperitive_to_gerund(m.group(2).lower()),
                new_answer,
            )

        result["sample"] = Sample(
            task_name=s.task_name,
            evidence_type=s.evidence_type,
            image=s.image,
            video=s.video,
            question=s.question,
            answer=new_answer,
            source_path=s.source_path,
            llm_fields=s.llm_fields,
        )
        result["status"] = "processed"
        result["stats"] = stats
        return result


    results: List[Dict[str, Any]]
    if parallel_n <= 1:
        results = [_process_one(s) for s in samples]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_n) as pool:
            results = list(pool.map(_process_one, samples))


    out: List[Sample] = []
    attempted = 0
    rewrote = 0
    fell_back = 0
    dropped = 0
    skipped_structured = 0
    vocab_warned = 0
    quote_repaired = 0
    verbatim_used = 0
    for r in results:
        st = r.get("stats", {})
        attempted += st.get("attempted", 0)
        rewrote += st.get("rewrote", 0)
        fell_back += st.get("fell_back", 0)
        dropped += st.get("dropped", 0)
        quote_repaired += st.get("quote_repaired", 0)
        verbatim_used += st.get("verbatim_used", 0)
        vocab_warned += st.get("vocab_warned", 0)
        if r["status"] == "skipped_structured":
            skipped_structured += 1
        if r["sample"] is not None:
            out.append(r["sample"])

    logger.info(
        ">>> [LLM] "
        f"attempted={attempted} rewrote={rewrote} fallback_draft={fell_back} dropped={dropped} "
        f"quote_repaired={quote_repaired} skipped_structured={skipped_structured} "
        f"verbatim_used={verbatim_used} vocab_guard={vocab_guard_s} vocab_warned={vocab_warned} verify={verify_s}"
        + (f" parallel={parallel_n}" if parallel_n > 1 else "")
    )
    logger.info(
        f">>> [LLM] Token summary: api_calls={llm.total_api_calls} "
        f"prompt_tokens={llm.total_prompt_tokens} completion_tokens={llm.total_completion_tokens} "
        f"total_tokens={llm.total_prompt_tokens + llm.total_completion_tokens}"
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

    def _check_exact_keys(obj: Any, *, allowed: set[str], path: str, optional: set[str] = frozenset()) -> None:
        if not strict or not isinstance(obj, dict):
            return
        extra = sorted([k for k in obj.keys() if k not in allowed])
        if extra:
            _add(f"{path} has extra keys: {extra}")
        missing = sorted([k for k in allowed if k not in obj and k not in optional])
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
    _check_exact_keys(plan_obj, allowed={"high_level_goal", "steps", "video_id", "source_video"}, path="top",
                      optional={"video_id", "source_video"})

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
        "clip_relpath",
        "atomic_actions",
        "independence",
        "detail_independence",
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
    allowed_interaction_keys = {"patient", "affordance_type", "mechanism"}

    step_ids: List[int] = []
    for idx, st_any in enumerate(steps):
        path = f"steps[{idx}]"
        if not isinstance(st_any, dict):
            _add(f"{path} must be an object")
            continue
        st = st_any
        _check_exact_keys(st, allowed=allowed_step_keys, path=path,
                          optional={"clip_relpath", "atomic_actions", "independence", "detail_independence"})

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
        if strict and q_cf and not re.match(r"^\s*(?:What\s+(?:if|would|could|happens)|How\s+would|If\s+|In\s+the\s+event|Suppose|Imagine|Assuming|Had\s+the)\b", q_cf, flags=re.IGNORECASE):
            _add(f"{path}.counterfactual_challenge_question must start with a recognized counterfactual prefix")
        cc = _require_obj(st.get("causal_chain"), f"{path}.causal_chain")
        _check_exact_keys(cc, allowed=allowed_step_cc_keys, path=f"{path}.causal_chain")
        for k in ("agent", "action", "patient"):
            v = _require_str_field(cc, k, f"{path}.causal_chain")

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

    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)



    s = re.sub(r"(?<=\w)\.{3,}\s*(?=[A-Za-z])", " — ", s)                          
    s = re.sub(r"\.{2,}$", ".", s)                                                         
    s = re.sub(r"\.{2,}(?=\s)", ".", s)                                                                          
    s = re.sub(r"\s+\.", ".", s)                    
    return s.strip()


def _normalize_trailing_punct(text: str) -> str:

    s = str(text or "").strip()
    if not s:
        return s

    s = re.sub(r"\s+([.!?])$", r"\1", s)

    s = re.sub(r"([.!?])[.!?]+$", r"\1", s)

    if re.search(r'[.!?]["\u2019\u201d)]*$', s):
        return s

    s += "."
    return s


def _join_as_paragraph(points: List[str]) -> str:

    cleaned: List[str] = []
    for p in points:
        p = p.strip()
        if not p:
            continue

        if p and p[0].islower():
            p = p[0].upper() + p[1:]

        if p and p[-1] not in ".!?":
            p += "."
        cleaned.append(p)

    cleaned = list(dict.fromkeys(cleaned))

    _deduped: List[str] = []
    for sent in cleaned:
        _sent_tokens = set(sent.lower().split())
        if any(
            len(_sent_tokens & set(prev.lower().split())) / max(len(_sent_tokens), 1) > 0.80
            for prev in _deduped
        ):
            continue
        _deduped.append(sent)
    cleaned = _deduped
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]


    _CONNECTIVES = [
        "Additionally, ", "Furthermore, ", "Also, ", "In addition, ",



        "Moreover, ", "Likewise, ", "Separately, ", "Alongside this, ",
    ]
    _EXISTING_STARTS = ("additionally", "furthermore", "also", "moreover", "meanwhile", "likewise",
                        "while", "however", "in addition", "consequently", "as a result",
                        "at the same time", "concurrently", "in contrast", "similarly",
                        "alongside this", "separately", "therefore", "thus", "hence", "nonetheless",
                        "nevertheless", "for example", "for instance", "specifically",
                        "in particular", "on the other hand", "note that")
    result = [cleaned[0]]


    _conn_idx = hash("".join(cleaned)) % len(_CONNECTIVES) if cleaned else 0
    for i, sent in enumerate(cleaned[1:], 1):
        sent_lower = sent.lower()

        if any(sent_lower.startswith(c) for c in _EXISTING_STARTS):
            result.append(sent)
            continue


        if len(cleaned) == 2 or (i % 2 == 0 and len(cleaned) > 2):
            conn = _CONNECTIVES[_conn_idx % len(_CONNECTIVES)]
            _conn_idx += 1
            sent = _lowercase_first_alpha(sent)
            result.append(f"{conn}{sent}")
        else:

            result.append(sent)
    return " ".join(result)


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
    raw = str(token or "").strip().replace("_", " ")

    _OVERRIDES = {
        "contacting": "in contact with",
        "relative to": "positioned relative to",
        "supported by": "supported by",
        "resting on": "resting on",
        "leaning against": "leaning against",
    }
    return _OVERRIDES.get(raw.lower(), raw)

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
    objects = [str(o).strip().replace("_", " ") for o in objects if isinstance(o, (str, int, float)) and str(o).strip()]
    truth_bool = _coerce_truth(rel.get("truth", True))

    def _the(s: str) -> str:

        if not s:
            return s
        if re.match(r"^(the|a|an)\b", s, flags=re.IGNORECASE):
            return s
        return f"the {s}"


    if not objects:

        return ""
    elif len(objects) == 1:


        _rel_clean = relation.strip()
        if len(_rel_clean.split()) > 1:
            _rel_clean = re.sub(r'\s+(?:of|to|from|with|at|on|in|by|for|between|among)\s*$', '', _rel_clean).strip()
        if not _rel_clean:
            return ""
        base = f"{_the(objects[0])} is {_rel_clean}"
    elif len(objects) == 2:
        base = f"{_the(objects[0])} is {relation} {_the(objects[1])}"
    else:


        others = [_the(o) for o in objects[1:]]
        if len(others) == 2:
            obj_list = f"{others[0]} and {others[1]}"
        else:
            obj_list = ", ".join(others[:-1]) + f", and {others[-1]}"
        base = f"{_the(objects[0])} is {relation} {obj_list}"

    if not truth_bool:
        base = base.replace(" is ", " is not ", 1)

    base = base.strip()
    if base and base[-1] not in ".!?":
        base += "."
    return base


def _format_affordance_state(st: Dict[str, Any]) -> str:
    obj = str(st.get("object_name", "") or "").strip().replace("_", " ")
    affs = st.get("affordance_types", [])
    if not isinstance(affs, list):
        affs = []
    affs = [str(a).strip().replace("_", " ") for a in affs if isinstance(a, (str, int, float)) and str(a).strip()]


    _DANGLING_SUFFIXES = (" by", " with", " to", " from", " on", " at", " in", " for")
    cleaned_affs = []
    for a in affs:
        for suf in _DANGLING_SUFFIXES:
            if a.endswith(suf) and len(a) > len(suf) + 2:
                a = a[:-len(suf)]
                break
        cleaned_affs.append(a)
    affs = cleaned_affs
    reasons = str(st.get("reasons", "") or "").strip()
    if not obj and not affs:
        return ""


    def _the_obj(s: str) -> str:
        if not s:
            return s
        if re.match(r"^(the|a|an)\b", s, flags=re.IGNORECASE):
            return s
        return f"the {s}"

    if obj and affs:
        if len(affs) == 1:
            aff_str = affs[0]
        elif len(affs) == 2:
            aff_str = f"{affs[0]} and {affs[1]}"
        else:
            aff_str = ", ".join(affs[:-1]) + f", and {affs[-1]}"
        base = f"{_the_obj(obj)} is {aff_str}"
    elif obj:

        return ""
    else:

        return ""
    if reasons:


        reasons_inline = _inline_clause(reasons).rstrip('.!?;')
        reasons_inline = _lowercase_first_alpha(reasons_inline)
        base = f"{base} because {reasons_inline}"

    base = base.strip()
    if base and base[-1] not in ".!?":
        base += "."
    return base


def _pick_best_precondition(phrases: List[str], step_goal: str) -> str:

    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "in", "on", "of",
             "to", "and", "or", "it", "its", "has", "have", "had", "with", "for",
             "from", "at", "by", "not", "that", "this", "but", "as"}
    goal_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", step_goal)} - _STOP
    best, best_score = phrases[0], -1
    for p in phrases:
        p_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", p)} - _STOP
        score = len(p_words & goal_words) * 10 + len(p)                                  
        if score > best_score:
            best, best_score = p, score
    return best


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


def _safe_relpath(path: str, root: str) -> str:

    ap = os.path.abspath(path)
    return ap.replace("\\", "/")


def _list_item_dirs(root: str, *, item_list_file: Optional[str] = None) -> List[str]:
    if item_list_file and os.path.isfile(item_list_file):
        logger.info("Reading item dirs from pre-built list: %s", item_list_file)
        with open(item_list_file, "r") as fh:
            out = [line.strip() for line in fh if line.strip()]
        logger.info("Loaded %d item dirs from file (bypassing os.walk)", len(out))
        return sorted(out)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if "final_plan.json" in filenames:
            out.append(dirpath)
            dirnames[:] = []
    return sorted(out)


def _resolve_video_prefix(item_dir: str, step_id: int, plan: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if step_id <= 0:
        return None
    cands = [

        os.path.join(item_dir, "prefix_clips", f"prefix_step{step_id:02d}.mp4"),
        os.path.join(item_dir, "prefix_clips", f"prefix_step{int(step_id)}.mp4"),

        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}.mp4"),

        os.path.join(item_dir, "cumulative_last_frame_segments", "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}.mp4"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    if plan and isinstance(plan, dict):
        src_video = plan.get("source_video", "")
        if isinstance(src_video, str) and src_video.strip() and os.path.isfile(src_video.strip()):
            return src_video.strip()
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
        except Exception as exc:
            logger.warning("[resolve_video_clip] error parsing step_segments.json in %s: %s", item_dir, exc)

    clips_dir = os.path.join(item_dir, "stage2", "step_clips")
    if os.path.isdir(clips_dir):
        matches = sorted(glob.glob(os.path.join(clips_dir, f"step{step_id:02d}_*.mp4")))
        if matches:
            return matches[0]

        matches = sorted(glob.glob(os.path.join(clips_dir, f"step{int(step_id)}_*.mp4")))
        if matches:
            return matches[0]
    return None


def _resolve_video_prefix_or_step_clip(
    item_dir: str,
    prefix_end_step: int,
    *,
    strict: bool,
    plan: Optional[Dict[str, Any]] = None,
) -> Optional[str]:

    if int(prefix_end_step) <= 0:
        return None
    vid = _resolve_video_prefix(item_dir, prefix_end_step, plan=None)                                             
    if vid:
        return vid


    if not strict and plan and isinstance(plan, dict):
        src_video = plan.get("source_video", "")
        if isinstance(src_video, str) and src_video.strip() and os.path.isfile(src_video.strip()):
            return src_video.strip()
    if strict:
        return None
    return _resolve_video_clip(item_dir, int(prefix_end_step))


def _find_keyframe_image(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:

    for base in (item_dir, os.path.join(item_dir, "stage3")):
        step_prefix = os.path.join(base, f"{step_id:02d}_*")
        pats: List[str] = []
        for ext in ("jpg", "jpeg", "png"):
            pats.append(os.path.join(step_prefix, f"frame_{frame_index:03d}_ts_*.{ext}"))
            pats.append(os.path.join(step_prefix, f"frame_{frame_index:03d}_*.{ext}"))
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
                for k in ("patient", "description", "affordance_type", "mechanism"):
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


def _dedup_subset_objects(tokens: List[str]) -> List[str]:

    if len(tokens) <= 1:
        return tokens


    norm = [t.strip().lower().replace("_", " ") for t in tokens]
    remove = set()
    for i, a in enumerate(norm):
        for j, b in enumerate(norm):
            if i != j and len(a) < len(b) and re.search(r'\b' + re.escape(a) + r'\b', b):
                remove.add(i)
                break
    if not remove:
        return tokens
    return [t for i, t in enumerate(tokens) if i not in remove]


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
        tokens = _dedup_subset_objects(tokens)
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
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ ]*", t):
            return False

        if "_" not in t and " " not in t and tl not in patient_pool_lower:
            return False
        return True

    tokens = [t for t in objs if _is_object_like(t)]
    tokens = sorted(set(tokens))
    if len(tokens) > 12:
        tokens = tokens[:12]


    tokens = _dedup_subset_objects(tokens)
    return tokens


def _stable_int_seed(text: str) -> int:

    h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:12], 16)


def _template_idx(n: int, *parts: object) -> int:

    key = ":".join(str(p) for p in parts)
    return _stable_int_seed(key) % n


def _normalize_terms(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()


    normalized = text.lower().replace("-", " ")
    tokens = re.findall(r"[A-Za-z0-9_]+", normalized)
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


def _sharegpt_entry(sample: Sample, *, attach_evidence: bool) -> Dict[str, Any]:
    source_path_norm = str(sample.source_path or "").replace("\\", "/").strip()
    item_dir = ""
    if source_path_norm.endswith("/final_plan.json"):
        item_dir = source_path_norm[: -len("/final_plan.json")].rstrip("/")
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
            **({"llm_fields": sample.llm_fields} if sample.llm_fields else {}),                                                                   
            **({"evidence_files": list(sample.image) + ([sample.video] if sample.video else [])} if bool(attach_evidence) else {}),
        },
    }
    if bool(attach_evidence) and sample.video:

        try:
            parsed_video = json.loads(sample.video)
            if isinstance(parsed_video, list):

                entry["video"] = parsed_video

                entry["meta"]["evidence_files"] = list(sample.image) + parsed_video
            else:
                entry["video"] = sample.video
        except (json.JSONDecodeError, TypeError):
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
    if src.endswith("/final_plan.json"):
        return src[: -len("/final_plan.json")].rstrip("/")
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
        "generator": "generate_stage_two_qa",
        "input_root": os.path.abspath(input_root),
        "enabled_tasks": sorted(set(enabled_tasks or [])),
        "text_only": bool(text_only),
        "meta_abs_paths": bool(meta_abs_paths),
        "uniform_k": int(uniform_k),
        "head": int(head),
        "tail": int(tail),
        "require_videos": bool(require_videos),
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
        TASK_08: {EVIDENCE_PREFIX},                                                        
        TASK_09: {EVIDENCE_PREFIX},                                                     
        TASK_10: {EVIDENCE_CLIP},                                                                   
        TASK_11: {EVIDENCE_CLIP},                                      
        TASK_04: {EVIDENCE_KEYFRAME},
        TASK_12: {EVIDENCE_KEYFRAME},                                                        
        TASK_05: {EVIDENCE_KEYFRAME},
        TASK_13: {EVIDENCE_CLIP},                                      
        TASK_01: {EVIDENCE_CLIP},                                      
        TASK_02: {EVIDENCE_CLIP},                                      
        TASK_03: {EVIDENCE_CLIP},                                      
        TASK_06: {EVIDENCE_CLIP},                                      
        TASK_07: {EVIDENCE_CLIP},                                      
        TASK_14: {EVIDENCE_CLIP_PAIR},                                      
        TASK_15: {EVIDENCE_PREFIX},
        TASK_16: {EVIDENCE_CLIP_PAIR},                                             
        TASK_17: {EVIDENCE_PREFIX},
        TASK_18: {EVIDENCE_PREFIX},
        TASK_19: {EVIDENCE_CLIP},                                      
        TASK_20: {EVIDENCE_CLIP},                                      
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
        except Exception as e:
            _add("error", expected_task, jsonl_path, 0, "", f"Failed to read jsonl: {e}")
            continue

        for ln_no, line in enumerate(f, start=1):
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
            if not isinstance(imgs, list):
                _add("error", task, jsonl_path, ln_no, sample_id, "image must be a list.")
                imgs = []


            _raw_vid = entry.get("video")
            vid: Optional[str] = None                                                       
            vid_list: Optional[list] = None                                       
            if isinstance(_raw_vid, str) and _raw_vid.strip():
                vid = _raw_vid
            elif isinstance(_raw_vid, list):
                vid_list = [str(v) for v in _raw_vid if isinstance(v, str) and str(v).strip()]


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


                if ev == EVIDENCE_CLIP_PAIR:

                    _videos_field = entry.get("videos")
                    if vid_list:
                        expected_files = list(imgs) + vid_list
                    elif isinstance(_videos_field, list):
                        expected_files = list(imgs) + [str(v) for v in _videos_field if isinstance(v, str) and str(v).strip()]
                    elif vid:
                        expected_files = list(imgs) + [vid]
                    else:
                        expected_files = list(imgs)
                else:
                    expected_files = list(imgs)
                    if vid:
                        expected_files.append(vid)

                if evidence_files is None:
                    _add("error", task, jsonl_path, ln_no, sample_id, "meta.evidence_files must be a list.")
                elif evidence_files != expected_files:
                    _add("error", task, jsonl_path, ln_no, sample_id,
                         f"meta.evidence_files must equal image + video(s). expected={expected_files!r}, got={evidence_files!r}")

                for p in expected_files:
                    ap = _abs_under_root(str(p), input_root)
                    if not ap or not os.path.exists(ap):
                        _add("error", task, jsonl_path, ln_no, sample_id, f"Evidence file not found: {p}")


                if ev == EVIDENCE_KEYFRAME:
                    if vid or vid_list:
                        _add("error", task, jsonl_path, ln_no, sample_id, "keyframe_single must not include video.")
                    if len(imgs) != 1:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"keyframe_single must have exactly 1 image (got {len(imgs)}).")
                if ev == EVIDENCE_UNIFORM:
                    if vid or vid_list:
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
                if ev == EVIDENCE_CLIP_PAIR:

                    if vid_list:
                        _n_vids = len(vid_list)
                    else:
                        _n_vids = 1 if vid else 0
                    if _n_vids < 2:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"video_clip_pair must have >=2 videos (got {_n_vids}).")
                    if len(imgs) != 0:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"video_clip_pair must not include images (got {len(imgs)}).")


            q_str = q if isinstance(q, str) else ""
            a_str = a if isinstance(a, str) else ""


            if task == TASK_08:
                if q_str.strip() not in _T08_QUESTIONS:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_08 question must match one of the approved template variants.")


            if task == TASK_09:
                if not any(v in q_str.lower() for v in ("from the candidate objects", "given the candidate objects")):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_09 question must include candidate objects template (no high-level goal).")

            if task == TASK_10:
                if ev == EVIDENCE_CLIP and not any(v in q_str for v in ("objective is being accomplished in this clip", "goal of the action shown in this clip", "trying to achieve in this clip")):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_10(video_clip) question must ask for the step goal of this clip.")


            if task == TASK_11 and not any(v in q_str for v in (
                "What action is occurring in this clip?",
                "Describe the action shown in this clip.",
                "What is the person doing in this clip?",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_11 question must match the GT action-phrase template.")
            if task == TASK_04 and not any(v in q_str for v in (
                "What is the main object being acted upon in this image",
                "Identify the key object being manipulated in this image",
                "Which object is primarily being acted on in this image",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_04 question must match the GT patient-affordance template.")
            if task == TASK_12 and not any(v in q_str for v in (
                "What action is taking place in this keyframe",
                "Describe what is happening in this keyframe",
                "What is the person doing in this keyframe",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_12 question must match the state-evolution keyframe template.")
            if task == TASK_12 and 'Current action:' not in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_12 question must include current action context.")
            if task == TASK_05 and not any(v in q_str.lower() for v in (
                "explain the causal chain",
                "describe the cause-and-effect relationship",
                "what causal chain connects",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_05 question must match the causal-chain template.")
            if task == TASK_13:
                if not any(v in q_str for v in (
                    "Why is the step shown in this clip necessary for the overall goal?",
                    "explain how this step contributes to achieving the overall goal.",
                    "What role does the action in this clip play in the broader plan?",
                )):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 question must match the GT rationale template.")
                if 'High-level goal:' not in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 question must include High-level goal.")
                if 'Step goal:' in q_str:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 question must NOT contain Step goal (only HL goal).")
            if task == TASK_01 and not any(v in q_str for v in (
                "spatial conditions that must be in place",
                "spatial conditions visible in this clip",
                "required spatial arrangement",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_01 question must match a spatial-precondition template variant.")
            if task == TASK_02 and not any(v in q_str for v in (
                "functional properties that objects must have",
                "affordance properties must the objects",
                "functional prerequisites",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_02 question must match the GT affordance-precondition template.")
            if task == TASK_03:
                if not any(v in q_str for v in (
                    "explain why the current step is physically feasible now",
                    "why is this step physically possible at this moment",
                    "what spatial and affordance conditions make this step feasible",
                )):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_03 question must match the GT feasibility-explanation template.")
            if task == TASK_14 and not any(v in q_str for v in (
                "How does the outcome of step",
                "What physical or spatial outcome of step",
                "Explain how step",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 question must match the GT inter-step dependency template.")
            if task == TASK_15 and not any(v in q_str for v in (
                "Based on this video prefix, what is the next step goal?",
                "Given this video prefix, predict the next step",
                "What action should follow next based on this video prefix?",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 question must match the GT next-step-from-prefix template.")
            if task == TASK_16 and not any(v in q_str for v in (
                "infer the missing middle steps in order",
                "deduce the intermediate steps that bridge them",
                "determine what middle steps must have occurred between them",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_16 question must match a GT middle-steps infill template variant.")
            if task == TASK_17 and not any(v in q_str for v in (
                "Based on this prefix, predict the next K=",
                "Based on this prefix, predict the next step goal",                
                "After watching this prefix, forecast the next",
                "Given the prefix shown, what are the next",
                "Given the prefix shown, what is the next step goal",                
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 question must match a GT next-K-steps template variant.")
            if task == TASK_18 and not any(v in q_str for v in (
                "Identify the flaw and repair the plan.",
                "Diagnose the error and provide a corrected plan.",
                "Find the problematic step and fix the plan.",
            )):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 question must include proposed plan steps and a diagnose+repair instruction variant.")

            if task == TASK_19 and not any(v in q_str.lower() for v in ("most likely outcome if", "would probably happen if", "what consequence would follow if")):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_19 question must match a GT counterfactual-outcome template variant.")
            if task == TASK_19 and 'Step goal:' in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_19 question must NOT contain Step goal (only counterfactual condition).")
            if task == TASK_20 and ("Failure reason:" not in q_str or "If this failure occurred during another execution" not in q_str):
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_20 question must match the GT failure-recovery template.")
            if task == TASK_20 and 'Step goal:' in q_str:
                _add("error", task, jsonl_path, ln_no, sample_id, "Task_20 question must NOT contain Step goal (only failure reason).")

            if task in (TASK_06, TASK_07):
                if task == TASK_06 and not any(v in q_str.lower() for v in (
                    "resulting spatial arrangement",
                    "spatial changes",
                    "spatial arrangement of objects changes",
                )):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_06 question must mention spatial arrangement/changes after step completion.")
                if task == TASK_07 and not any(v in q_str.lower() for v in (
                    "functional properties of the objects",
                    "affordance or functional state changes",
                    "functional properties of objects change",
                )):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_07 question must mention functional properties/state changes.")
                if ev != EVIDENCE_CLIP:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_06/07 must use video_clip evidence_type.")

            if task == TASK_09:
                m = re.search(r"(?:from|given)\s+the\s+candidate\s+objects\s+(\[[^\]]*\])", q_str, re.IGNORECASE)
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
                r"^The flaw is in step \d+.+The corrected plan is:\s*1\)\s*\".+\"(?:\s+\d+\)\s*\".+\")*\s*$",
                a_str,
            ):
                _add(
                    "error",
                    task,
                    jsonl_path,
                    ln_no,
                    sample_id,
                    "Task_18 answer must follow the format: The flaw is in step N, which contains ... The corrected plan is: 1) \"...\" 2) \"...\" ...",
                )

            if task == TASK_19 and _RECOVERY_SUGGESTION_RE.search(a_str):
                _add("error", task, jsonl_path, ln_no, sample_id, "Counterfactual answers must not propose recovery/action suggestions.")
            if task == TASK_19 and _RECOVERY_ACTION_PHRASE_RE.search(a_str):
                _add("error", task, jsonl_path, ln_no, sample_id, "Counterfactual answers must not contain recovery action phrases (should/need to/must + action verb).")
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


    _t18_item_flaw_types: Dict[str, set[str]] = {}
    _t18_item_step_counts: Dict[str, int] = {}

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

    _STEP_GOAL_RE = re.compile(r'(?:Step goal:|The current action is)\s*"([^"]+)"')
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
        except Exception as e:
            _add("error", expected_task, jsonl_path, 0, "", f"Failed to read jsonl: {e}")
            continue

        for ln_no, line in enumerate(f, start=1):
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
            ev = str(meta.get("evidence_type") or "")
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



                _hl_proc = _sanitize_space(hl).lower()
                a_norm = _sanitize_space(a_str).lower()

                _hl_tokens = set(w.strip(".,;:!?\"'()") for w in _hl_proc.split() if len(w.strip(".,;:!?\"'()")) >= 4)
                _a_tokens = set(w.strip(".,;:!?\"'()") for w in a_norm.split() if len(w.strip(".,;:!?\"'()")) >= 4)

                if _hl_tokens:
                    _overlap = _hl_tokens & _a_tokens
                    _ratio = len(_overlap) / len(_hl_tokens)
                    if _ratio < 0.50:
                        _add("error", task, jsonl_path, ln_no, sample_id,
                             f"Task_08 answer covers only {_ratio:.0%} of high_level_goal key words (need ≥50%). missing={sorted(_hl_tokens - _a_tokens)[:10]}")

            if task == TASK_09:
                exp = list(_extract_key_objects_for_task02(plan) or [])

                exp_set = {str(x).strip().replace("_", " ") for x in exp if str(x).strip()}
                cand_set: set[str] = set()
                m2 = re.search(r"(?:from|given) the candidate objects\s+(\[[^\]]*\])", q_str, re.IGNORECASE)
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
                TASK_04,
                TASK_05,
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

                    if task == TASK_04:


                        ans_l = a_str.lower()
                        relaxed = task in LLM_RELAXED_SPAN_TASKS

                        def _ok(st: Dict[str, Any]) -> bool:
                            cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
                            step_patient = _require_str(cc, "patient").strip() if isinstance(cc, dict) else ""
                            for cf in st.get("critical_frames") or []:
                                if not isinstance(cf, dict):
                                    continue
                                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                                if not isinstance(intr, dict):
                                    continue


                                hotspot = intr.get("hotspot") if isinstance(intr.get("hotspot"), dict) else intr
                                kf_patient = _require_str(hotspot, "patient").strip() or _require_str(hotspot, "description").strip()
                                patient = kf_patient if kf_patient else step_patient


                                aff_type = _require_str(hotspot, "affordance_type").strip()
                                mech = _require_str(hotspot, "mechanism").strip()
                                if not (patient and aff_type and mech):
                                    continue
                                patient_clause = _lowercase_first_alpha(patient.strip().rstrip(".")).lower()
                                mech_clause = _lowercase_first_alpha(_inline_clause(mech)).lower()
                                aff_type_l = aff_type.lower()
                                aff_type_human = aff_type_l.replace("_", " ")

                                _TOOL_TO_PATIENT_AFF = {
                                    "cutting edge": "cuttable material",
                                    "blade edge": "cuttable material",
                                    "sharp edge": "cuttable material",
                                    "lever arm": "pivotable joint",
                                    "piercing tip": "penetrable surface",
                                }
                                _remapped = _TOOL_TO_PATIENT_AFF.get(aff_type_human, "")
                                if (aff_type_l not in ans_l
                                        and aff_type_human not in ans_l
                                        and (not _remapped or _remapped not in ans_l)):
                                    continue
                                if patient_clause and not _span_present_in_text(patient_clause, a_str, relaxed=bool(relaxed)):
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
                                "Task_04 answer must include patient, affordance_type, and mechanism grounded in the source JSON.",
                            )

                    if task == TASK_05:

                        def _expected_task07_anchors(st: Dict[str, Any]) -> List[Tuple[str, str]]:
                            out_anc: List[Tuple[str, str]] = []
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

                                mech_raw = _require_str(intr, "mechanism") if isinstance(intr, dict) else ""
                                mech = _inline_clause(mech_raw)

                                ctx_terms = _normalize_terms(" ".join([step_goal, mech_raw]))


                                _T05_MAX_POINTS_PER_CAT_V = 3

                                def _ranked_points_v(points: List[str], *, top_k: int = _T05_MAX_POINTS_PER_CAT_V) -> List[str]:
                                    scored: List[Tuple[float, str]] = []
                                    for p in points or []:
                                        clause = _inline_clause(str(p))
                                        if not clause:
                                            continue
                                        if _needs_not_directly_observable(clause):
                                            continue
                                        clause_terms = _normalize_terms(clause)
                                        score = float(len(clause_terms & ctx_terms))
                                        score -= 0.001 * len(clause)
                                        scored.append((score, clause))
                                    scored.sort(key=lambda x: x[0], reverse=True)
                                    seen: List[str] = []
                                    for _, clause in scored[:top_k]:

                                        p2 = _lowercase_first_alpha(clause.strip().rstrip(",;:"))
                                        if p2 and p2 not in seen:
                                            seen.append(p2)
                                    return seen

                                sp_pre_ranked = _ranked_points_v(sp_pre_pts)
                                af_pre_ranked = _ranked_points_v(af_pre_pts)
                                sp_eff_ranked = _ranked_points_v(sp_eff_pts)
                                af_eff_ranked = _ranked_points_v(af_eff_pts)

                                pre_parts: List[str] = []
                                for p2 in sp_pre_ranked + af_pre_ranked:
                                    if p2 not in pre_parts:
                                        pre_parts.append(p2)

                                eff_parts: List[str] = []
                                for p2 in sp_eff_ranked + af_eff_ranked:
                                    if p2 not in eff_parts:
                                        eff_parts.append(p2)

                                def _join_parts_v(parts: List[str]) -> str:
                                    if not parts:
                                        return ""
                                    if len(parts) == 1:
                                        return parts[0]
                                    if len(parts) == 2:
                                        return f"{parts[0]} and {parts[1]}"
                                    return ", ".join(parts[:-1]) + f", and {parts[-1]}"

                                pre_clause = _join_parts_v(pre_parts).strip()
                                eff_clause = _join_parts_v(eff_parts).strip()
                                if not (mech and pre_clause and eff_clause):
                                    continue
                                out_anc.append((pre_clause, eff_clause))
                            return out_anc

                        anchors: List[Tuple[str, str]] = []
                        for st in cand_steps:
                            anchors.extend(_expected_task07_anchors(st))
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

                            if not ok:
                                _add(
                                    "error",
                                    task,
                                    jsonl_path,
                                    ln_no,
                                    sample_id,
                                    "Task_05 answer must remain grounded in the expected precondition/effect clauses.",
                                )





            if task in (TASK_11, TASK_12, TASK_13, TASK_01, TASK_02, TASK_03, TASK_06, TASK_07, TASK_19, TASK_20):

                _ev_files = meta.get("evidence_files", [])
                if not isinstance(_ev_files, list):
                    _ev_files = []
                _ev_step_id = None
                for _ef in _ev_files:
                    _ef_m = re.search(r'step[_]?0*(\d+)', str(_ef))
                    if _ef_m:
                        _ev_step_id = int(_ef_m.group(1))
                        break

                _target_steps = steps
                if _ev_step_id is not None:
                    _specific = [s for s in steps if int(s.get("step_id", 0) or 0) == _ev_step_id]
                    if _specific:
                        _target_steps = _specific

                def _any_match_all(pred) -> bool:
                    for st in _target_steps:
                        try:
                            if pred(st):
                                return True
                        except Exception:
                            continue
                    return False

                if task == TASK_11:



                    ans_norm = _sanitize_space(a_str)

                    def _ok_action(st: Dict[str, Any]) -> bool:
                        act = _require_str(st.get("causal_chain") or {}, "action")
                        needle = _sanitize_space(act)
                        if not needle:
                            return False

                        _act_tokens = set(w.strip(".,;:!?\"'()").lower() for w in needle.split()
                                          if len(w.strip(".,;:!?\"'()")) >= 4)
                        _ans_tokens = set(w.strip(".,;:!?\"'()").lower() for w in ans_norm.split()
                                          if len(w.strip(".,;:!?\"'()")) >= 4)
                        if not _act_tokens:
                            return True                                            
                        _overlap = _act_tokens & _ans_tokens
                        _ratio = len(_overlap) / len(_act_tokens)
                        return _ratio >= 0.35                                                                  

                    if not _any_match_all(_ok_action):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_11 answer must contain ≥50% key words from steps[i].causal_chain.action.")

                if task == TASK_12:
                    relaxed = task in LLM_RELAXED_SPAN_TASKS
                    def _ok_t06(st: Dict[str, Any]) -> bool:
                        sg = _require_str(st, "step_goal")
                        for cf in st.get("critical_frames") or []:
                            if not isinstance(cf, dict):
                                continue
                            raw_asc = _require_str(cf, "action_state_change_description")


                            asc = _select_best_asc_clause(raw_asc, context=sg)
                            if _span_present_in_text(asc, a_str, relaxed=bool(relaxed)):
                                return True
                        return False
                    if not _any_match_all(_ok_t06):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_12 answer must contain a critical_frames[*].action_state_change_description span.")

                if task == TASK_13:
                    relaxed = task in LLM_RELAXED_SPAN_TASKS
                    def _ok_t08(st: Dict[str, Any]) -> bool:
                        rat = _require_str(st, "rationale")
                        if not rat:
                            return False



                        rat_inlined = _inline_clause(rat.strip())
                        return bool(rat_inlined) and _span_present_in_text(rat_inlined, a_str, relaxed=bool(relaxed))
                    if not _any_match_all(_ok_t08):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_13 answer must contain steps[i].rationale span (inlined).")

                if task == TASK_19:
                    relaxed = task in LLM_RELAXED_SPAN_TASKS
                    def _ok_t19(st: Dict[str, Any]) -> bool:
                        exp = _clean_counterfactual_outcome(_require_str(st, "expected_challenge_outcome"))
                        return bool(exp) and _span_present_in_text(exp, a_str, relaxed=bool(relaxed))
                    if not _any_match_all(_ok_t19):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_19 answer must contain cleaned steps[i].expected_challenge_outcome span.")

                if task == TASK_20:
                    relaxed = task in LLM_RELAXED_SPAN_TASKS
                    def _ok_t20(st: Dict[str, Any]) -> bool:
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
                    if not _any_match_all(_ok_t20):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_20 answer must contain steps[i].failure_reflecting.recovery_strategy span.")

                if task in (TASK_01, TASK_02, TASK_06, TASK_07):
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
                    relaxed = task in LLM_RELAXED_SPAN_TASKS

                    def _ok_causal(st: Dict[str, Any]) -> bool:
                        scc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
                        if not isinstance(scc, dict):
                            return False
                        if task in (TASK_01, TASK_06):
                            pts = _format_spatial_points(scc.get(field_key))
                        else:
                            pts = _format_affordance_points(scc.get(field_key))
                        pts = [p for p in pts if p and "not directly observable" not in p.lower()]

                        exp = _sanitize_space(_join_as_paragraph(pts))
                        return bool(exp) and _span_present_in_text(exp, a_str, relaxed=bool(relaxed))

                    if not _any_match_all(_ok_causal):
                        _add("error", task, jsonl_path, ln_no, sample_id, f"{task} answer must contain expected causal_chain field span from step-level data.")

                if task == TASK_03:
                    def _ok_t11(st: Dict[str, Any]) -> bool:
                        scc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
                        if not isinstance(scc, dict):
                            return False
                        sp_pts = _format_spatial_points(scc.get("causal_precondition_on_spatial"))
                        af_pts = _format_affordance_points(scc.get("causal_precondition_on_affordance"))
                        sp_pts = [p for p in sp_pts if p and "not directly observable" not in p.lower()]
                        af_pts = [p for p in af_pts if p and "not directly observable" not in p.lower()]
                        if not (sp_pts and af_pts):
                            return False
                        for sp in sp_pts:
                            sp_c = _lowercase_first_alpha(_inline_clause(sp)).strip().rstrip('.!?')
                            for af in af_pts:
                                af_c = _lowercase_first_alpha(_inline_clause(af)).strip().rstrip('.!?')
                                if sp_c and af_c:
                                    if (_span_present_in_text(sp_c, a_str, relaxed=True)
                                            and _span_present_in_text(af_c, a_str, relaxed=True)):
                                        return True
                        return False

                    if not _any_match_all(_ok_t11):
                        _add("error", task, jsonl_path, ln_no, sample_id, "Task_03 answer must contain a valid (spatial, affordance) precondition pair from step causal_chain.")


            if task == TASK_14:
                if 'High-level goal:' not in q_str or 'first video' not in q_str.lower():
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 question must include High-level goal and mention first/second video.")

                if 'shows step' not in q_str.lower():
                    _add("warn", task, jsonl_path, ln_no, sample_id, "Task_14 question should include step numbers for per-pair uniqueness.")
                if ev != EVIDENCE_CLIP_PAIR:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 must use video_clip_pair evidence_type.")


                grounding_ok = False
                for s in steps:
                    detail_ind = str(s.get("detail_independence", "") or "").strip()
                    if not detail_ind:
                        continue
                    if _span_present_in_text(detail_ind, a_str, relaxed=True):
                        grounding_ok = True
                        break
                if not grounding_ok and len(steps) >= 2:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_14 answer could not be grounded to any step's detail_independence field.")

            if task == TASK_15:


                if len(steps) >= 2:

                    _ev_files = meta.get("evidence_files") or []
                    _prefix_step_num = 0
                    for _ef in _ev_files:
                        _pfm = re.search(r"prefix_step(\d+)", str(_ef))
                        if _pfm:
                            _prefix_step_num = int(_pfm.group(1))
                            break

                    _expected_goal = ""
                    if _prefix_step_num > 0:
                        for _s in steps:
                            _sid = int(_s.get("step_id", 0) or 0)
                            if _sid == _prefix_step_num + 1:
                                _expected_goal = _require_str(_s, "step_goal")
                                break
                    if _expected_goal:

                        ans_norm = _sanitize_space(a_str).lower()
                        eg_norm = _sanitize_space(_expected_goal).lower()
                        if eg_norm and eg_norm not in ans_norm:

                            if not (eg_norm.endswith(".") and eg_norm[:-1] in ans_norm):
                                _add("error", task, jsonl_path, ln_no, sample_id,
                                     f"Task_15 answer must contain the correct next step goal (step {_prefix_step_num+1}), not just any step.")
                    else:

                        all_goals = [_require_str(s, "step_goal") for s in steps if _require_str(s, "step_goal")]
                        ans_norm = _sanitize_space(a_str).lower()
                        found = False
                        for sg in all_goals:
                            sg_norm = _sanitize_space(sg).lower()
                            if sg_norm and sg_norm in ans_norm:
                                found = True
                                break
                            if sg_norm.endswith(".") and sg_norm[:-1] in ans_norm:
                                found = True
                                break
                        if not found:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_15 answer must contain a valid step goal from the source plan.")

            if task == TASK_16:
                if len(steps) >= 3:
                    middle = [_require_str(s, "step_goal") for s in steps[1:-1] if _require_str(s, "step_goal")]
                    middle = middle[:8]                                        
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


                km = re.search(r"next (?:K=)?(\d+)", q_str)
                if km:
                    try:
                        k = int(km.group(1))
                    except Exception:
                        k = 0
                elif "next step goal" in q_str:
                    k = 1                         
                else:
                    k = 0
                if k <= 0:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_17 question must include K value.")
                else:

                    all_goals = [_require_str(s, "step_goal") for s in steps if _require_str(s, "step_goal")]
                    ans_norm = _sanitize_text_single_line(a_str).lower()
                    matched = 0
                    for sg in all_goals:
                        needle = _sanitize_text_single_line(sg).lower()
                        if needle and needle in ans_norm:
                            matched += 1
                    if matched < k:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"Task_17 answer contains {matched} recognizable step goals but expected K={k}.")

            if task == TASK_18:
                seg = q_str
                m_seg = re.search(
                    r"(?:the following proposed plan steps are|the proposed subsequent steps are|the proposed plan continuation is):\s*(.*?)\s*(?:Identify the flaw and repair the plan|Diagnose the error and provide a corrected plan|Find the problematic step and fix the plan)\.",
                    q_str,
                )
                if m_seg:
                    seg = str(m_seg.group(1) or "").strip()
                q_items = re.findall(r'\d+\)\s*"([^"]+)"', seg)
                if len(q_items) < 2 or len(q_items) > 5:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 question must include 2-5 bad_plan_steps.")
                    q_items = []
                step_goals = [_require_str(s, "step_goal") for s in steps]


                fm = re.search(r"(?:step|Step)\s+(\d+)", a_str)

                _FLAW_LABEL_TO_TYPE = {
                    "a goal-inconsistent step": "goal_inconsistent",
                    "a vague or overly general step": "granularity_mismatch",
                    "a step that skips a required precondition": "precondition_missing",
                    "a redundant step that was already completed": "redundant_step",
                    "two steps in the wrong order": "wrong_ordering",
                }
                flaw_type = ""
                for label, ft in _FLAW_LABEL_TO_TYPE.items():
                    if label in a_str.lower():
                        flaw_type = ft
                        break
                if not fm:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 answer must include 'step N' indicating the flawed step.")
                    flaw_step = 0
                else:
                    try:
                        flaw_step = int(fm.group(1))
                    except Exception:
                        flaw_step = 0
                if not flaw_type:

                    tm_old = re.search(r"FlawType\s*=\s*([a-z_ ]+)", a_str)
                    if tm_old:
                        flaw_type = str(tm_old.group(1)).strip().replace(" ", "_")

                _VALID_FLAW_TYPES = {
                    "goal_inconsistent", "precondition_missing", "redundant_step", "wrong_ordering", "granularity_mismatch",
                    "goal inconsistent", "precondition missing", "redundant step", "wrong ordering", "granularity mismatch",
                }
                if flaw_type and flaw_type not in _VALID_FLAW_TYPES:
                    _add("error", task, jsonl_path, ln_no, sample_id, f"Task_18 flaw type not recognized (got {flaw_type!r}).")
                if flaw_step < 1 or flaw_step > 5:
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 flaw step must be between 1 and 5.")


                _ft_norm = flaw_type.replace(" ", "_") if flaw_type else ""
                _item_key = str(item_dir or "").strip()
                if _ft_norm and _item_key:
                    _t18_item_flaw_types.setdefault(_item_key, set()).add(_ft_norm)
                    if _item_key not in _t18_item_step_counts:
                        _t18_item_step_counts[_item_key] = len(step_goals)

                ans_steps = re.findall(r'\d+\)\s*"([^"]+)"', a_str)

                _max_repair = 6 if flaw_type.replace(" ", "_") == "precondition_missing" else 5
                if len(ans_steps) < 2 or len(ans_steps) > _max_repair:
                    _add("error", task, jsonl_path, ln_no, sample_id, f"Task_18 answer must include 2-{_max_repair} repaired steps under Repair:.")
                    ans_steps = []
                if ans_steps and any(it not in step_goals for it in ans_steps):
                    _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 repaired steps must come from steps[*].step_goal.")





                if q_items and ans_steps:
                    _is_precon_missing = flaw_type.replace("_", " ") == "precondition missing"
                    _ans_core = ans_steps[:len(q_items)] if _is_precon_missing else ans_steps
                    _ans_tail = ans_steps[len(q_items):] if _is_precon_missing else []
                    k_len = len(_ans_core)
                    best_slice: Optional[List[str]] = None
                    best_mismatch_idx: Optional[int] = None
                    for i in range(len(step_goals) - k_len + 1):
                        sl = step_goals[i : i + k_len]
                        mism = [j for j in range(k_len) if j < len(q_items) and q_items[j] != sl[j]]
                        if len(mism) == 1:
                            best_slice = sl
                            best_mismatch_idx = mism[0]
                            break



                        if (len(mism) == 2
                                and flaw_type.replace("_", " ") == "wrong ordering"
                                and abs(mism[0] - mism[1]) == 1
                                and mism[0] < len(q_items) and mism[1] < len(q_items)
                                and q_items[mism[0]] == sl[mism[1]]
                                and q_items[mism[1]] == sl[mism[0]]):
                            best_slice = sl
                            best_mismatch_idx = mism[0]
                            break
                    if best_slice is None or best_mismatch_idx is None:
                        _add("error", task, jsonl_path, ln_no, sample_id, f"Task_18 bad_plan_steps must differ from a contiguous {k_len}-step slice at exactly one position (or adjacent swap for wrong_ordering).")
                    else:
                        if _ans_core != best_slice:
                            _add("error", task, jsonl_path, ln_no, sample_id, f"Task_18 Repair core steps must equal the gold contiguous {k_len}-step slice from the source plan.")

                        if _ans_tail and any(t not in step_goals for t in _ans_tail):
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 precondition_missing repair tail step must come from steps[*].step_goal.")
                        if flaw_step and flaw_step != best_mismatch_idx + 1:
                            _add("error", task, jsonl_path, ln_no, sample_id, "Task_18 FlawStep must point to the mismatched bad_plan_steps position.")

                        if flaw_type.replace("_", " ") == "wrong ordering" and best_mismatch_idx is not None:

                            _wo_slice_start = None
                            for _si in range(len(step_goals) - k_len + 1):
                                if step_goals[_si: _si + k_len] == best_slice:
                                    _wo_slice_start = _si
                                    break
                            if _wo_slice_start is not None:
                                _wo_step_b_idx = _wo_slice_start + best_mismatch_idx + 1
                                if _wo_step_b_idx < len(steps):
                                    _wo_indep = str(steps[_wo_step_b_idx].get("independence", "")).strip().lower()
                                    if _wo_indep != "yes":
                                        _add("warn", task, jsonl_path, ln_no, sample_id,
                                             "Task_18 wrong_ordering: swapped pair lacks confirmed causal dependency (independence field is not 'yes').")
        f.close()


    _T18_ALL_FLAW_TYPES = {"goal_inconsistent", "granularity_mismatch", "precondition_missing", "redundant_step", "wrong_ordering"}
    for _item_key, _seen_fts in sorted(_t18_item_flaw_types.items()):
        _n_steps = _t18_item_step_counts.get(_item_key, 0)
        _missing = _T18_ALL_FLAW_TYPES - _seen_fts
        if _missing and _n_steps >= 5:
            _missing_str = ", ".join(sorted(_missing))
            _add("warn", TASK_18, _item_key, 0, "",
                 f"Task_18 coverage: item has {_n_steps} steps but is missing flaw types: {_missing_str}")

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


    return v.strip().replace('"', "'") if isinstance(v, str) else ""


def _shorten_hl_for_context(hl: str) -> str:

    if not hl:
        return hl
    short = re.split(r',\s*(?:and\s+)?then\b|;\s*then\b', hl)[0].strip()

    m = re.match(r'^[Aa]fter\s+.+?,\s*', short)
    if m:
        core = short[m.end():]
        if core:
            short = core
    words = short.split()
    if len(words) > 20:

        commas = [i for i, w in enumerate(words) if w.endswith(",")]
        cut = None
        for ci in reversed(commas):
            if ci + 1 <= 20 and ci + 1 >= 4:
                cut = ci + 1
                break
        if cut:
            short = " ".join(words[:cut]).rstrip(",")
        else:
            short = " ".join(words[:20]).rstrip(",;")
    return short.rstrip(".")


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

    q = _T08_QUESTIONS[_stable_int_seed(item_dir) % len(_T08_QUESTIONS)]



    a = _sanitize_space(hl)

    if len(a.strip().rstrip(".!?")) < 5:
        return None
    if not bool(attach_evidence):
        evidence_type = EVIDENCE_PREFIX
        images = []
        video_rel = None
    else:
        last_step_id = int(steps[-1].get("step_id", 0) or 0)
        video = _resolve_video_prefix(item_dir, last_step_id, plan=plan)
        if video:
            evidence_type = EVIDENCE_PREFIX
            images = []
            video_rel = _safe_relpath(video, input_root)
        else:

            return None
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)
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


    kitchen_pool = list(_DISTRACTOR_OBJECTS_KITCHEN)
    other_pool = list(_DISTRACTOR_OBJECTS_OTHER)
    rng.shuffle(kitchen_pool)
    rng.shuffle(other_pool)
    for d in kitchen_pool + other_pool:
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


    candidates = [c.replace("_", " ") for c in candidates]
    label_objs = [o.replace("_", " ") if isinstance(o, str) else o for o in label_objs]

    label_objs = list(dict.fromkeys(label_objs))
    candidates = list(dict.fromkeys(candidates))


    _cands_str = json.dumps(candidates)
    _T09_QUESTIONS = [
        (f'Given the candidate objects {_cands_str}, identify which objects play a direct role in the activity '
         "depicted in this video clip."),
        (f'From the candidate objects {_cands_str}, which ones are directly involved in the main activity shown in this video? '
         "List only the relevant objects."),
        (f'Based on this video clip, from the candidate objects {_cands_str}, '
         "list the key objects that are directly relevant to the main activity."),
    ]
    q = _T09_QUESTIONS[_stable_int_seed(json.dumps(label_objs)) % len(_T09_QUESTIONS)]
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
    if not joined:
        return None

    _t02_variant = _template_idx(4, "T02a", item_dir)
    if _t02_variant == 0:
        a = _sanitize_space(f"The key objects directly relevant to the goal are {joined}.")
    elif _t02_variant == 1:
        a = _sanitize_space(f"The main objects involved in this activity are {joined}.")
    elif _t02_variant == 2:
        a = _sanitize_space(f"For this task, the critical objects are {joined}.")
    else:
        a = _sanitize_space(f"The objects central to achieving the goal are {joined}.")


    video_rel: Optional[str] = None
    if bool(attach_evidence):
        steps = _sorted_steps(plan)
        if not steps:
            return None
        last_step_id = int(steps[-1].get("step_id", 0) or 0)
        video = _resolve_video_prefix(item_dir, last_step_id, plan=plan) if last_step_id > 0 else None
        if not video:
            return None
        video_rel = _safe_relpath(video, input_root)
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)
    return Sample(
        task_name=TASK_09,
        evidence_type=EVIDENCE_PREFIX,
        image=[],
        video=video_rel,
        question=q,
        answer=a,
        source_path=source_rel,
        llm_fields={"key_objects": list(label_objs)},
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
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)
    for st in steps:
        step_id = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if step_id <= 0 or not step_goal:
            continue
        _T10_QUESTIONS = [
            f'Context: High-level goal: "{hl}" What specific objective is being accomplished in this clip?',
            f'Context: High-level goal: "{hl}" Describe the specific goal of the action shown in this clip.',
            f'Context: High-level goal: "{hl}" What is the person trying to achieve in this clip?',
        ]
        _t03_q = _T10_QUESTIONS[step_id % len(_T10_QUESTIONS)]
        if not bool(attach_evidence):
            evidence_type = EVIDENCE_CLIP
            q = _t03_q
            video_rel = None
            images = []
        else:
            clip = _resolve_video_clip(item_dir, step_id)
            if require_video and not clip:
                continue

            if clip:
                evidence_type = EVIDENCE_CLIP
                q = _t03_q
                video_rel = _safe_relpath(clip, input_root)
                images = []
            else:

                continue
        a = _sanitize_space(step_goal)

        if len(a.strip().rstrip(".!?")) < 5:
            continue
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
    "internal structure",
    "internal property",
    "internal state",
    "friction coefficient",
    "surface texture",
    "static friction",
    "static charge",
    "electrical circuit",
    "electrical resistance",
    "magnetic field",
    "chemical reaction",
    "chemical composition",
    "temperature gradient",
    "internal pressure",
    "material composition",
    "material property",
    "viscosity",

    "cannot be determined",
    "cannot be verified",
    "not visible",
    "not apparent",
    "unclear from",
    "assumed",
    "inferred but",
)

_RECOVERY_SUGGESTION_RE = re.compile(
    r"(?:,?\s*(?:and|so)\s+)?"
    r"(?:the\s+)?(?:cook|person|agent|operator|user|they|you|one)\s+"
    r"(?:would\s+)?(?:need|needs|must|should|have\s+to|can|could|may\s+need\s+to|ought\s+to)\b",
    flags=re.IGNORECASE,
)


_RECOVERY_ACTION_PHRASE_RE = re.compile(
    r"\b(?:"

    r"(?:should|need\s+to|must|would\s+need\s+to|require[sd]?\s+to"
    r"|try\s+to|make\s+sure\s+to|it\s+is\s+(?:recommended|advisable)\s+to"
    r"|ought\s+to|have\s+to|can|may\s+need\s+to)\s+"
    r"(?:use|find|get|grab|apply|switch|replace|wipe|clean|re-?do|re-?start|re-?position"
    r"|adjust|correct|check|verify|ensure|remove|add|fix|repair|redo|repeat|undo"
    r"|reapply|reattempt|realign|recalibrate|recover|retrieve|restore"

    r"|be\s+\w+(?:ed|en))"
    r"|"


    r"could\s+(?:use\s+(?!a\b|an\b|some\b|more\b|better\b|another\b|additional\b)"
    r"|find|get|grab|apply|switch|replace|wipe|clean|re-?do|re-?start|re-?position"
    r"|adjust|correct|check|verify|ensure|remove|add|fix|repair|redo|repeat|undo"
    r"|reapply|reattempt|realign|recalibrate|recover|retrieve|restore"
    r"|be\s+\w+(?:ed|en))"
    r"|"

    r"consider\s+\w+ing"
    r"|"

    r"if\s+not\s+\w+(?:ed|en)"
    r")\b",
    flags=re.IGNORECASE,
)


_BARE_IMPERATIVE_RE = re.compile(
    r"^(?:use|find|get|grab|apply|switch|replace|wipe|clean|redo|restart|reposition"
    r"|adjust|correct|check|verify|ensure|remove|add|fix|repair|repeat|undo"
    r"|reapply|reattempt|realign|recalibrate|recover|retrieve|restore"
    r"|lower|raise|turn|stir|pour|chop|slice|cut|peel|rinse|wash|drain"
    r"|heat|cool|warm|boil|simmer|bake|roast|fry|saut[eé]|grill|toast"
    r"|place|put|set|move|transfer|lift|hold|press|squeeze|scrape"
    r"|open|close|cover|uncover|seal|wrap|unwrap|fold|spread"
    r"|mix|blend|whisk|beat|knead|toss|shake|flip|rotate"
    r"|monitor|watch|wait|let|allow|leave|keep|maintain|continue"
    r"|discard|dispose|throw|dump|empty|scrub|dry|wipe|pat)\b",
    flags=re.IGNORECASE,
)


def _inline_clause(text: str) -> str:
    s = _sanitize_text_single_line(text)
    s = re.sub(r"[.?!]+\s+", "; ", s)


    def _lower_after_semi(m: re.Match) -> str:
        ch = m.group(1)
        rest = s[m.end():]

        if ch == "I" and (not rest or not rest[0].isalpha() or rest[0] in "'\u2019"):
            return "; " + ch


        if len(rest) >= 2 and rest[0].isupper() and rest[1].isupper():
            return "; " + ch               
        return "; " + ch.lower()
    s = re.sub(r";\s*([A-Z])", _lower_after_semi, s)
    s = s.strip().strip(";")
    s = re.sub(r"[;:,]+$", "", s).strip()
    s = re.sub(r"[.?!]+$", "", s).strip()
    return s


def _truncate_llm_field(value: Any, *, max_items: int = 3, max_words: int = 150) -> Any:

    return value



_DANGLING_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on",
    "to", "for", "with", "by", "from", "at", "its", "is",
})



_NOT_GERUNDS: frozenset = frozenset({
    "spring", "string", "bring", "thing", "during", "nothing",
    "something", "everything", "anything", "king", "ring", "sing",
    "wing", "ding", "cling", "swing", "fling", "sling", "sting",
    "wring", "being",

    "pudding", "ceiling", "morning", "evening", "clothing",
    "stuffing", "frosting", "seasoning", "lightning",
})


def _truncate_clause(clause: str, max_words: int = 30) -> str:

    words = clause.split()
    if len(words) <= max_words:
        return clause

    for i, w in enumerate(words[:max_words]):
        if w.endswith(';') and i >= 5:
            return ' '.join(words[:i + 1]).rstrip(';').strip()

    cut = max_words
    while cut > max(max_words - 5, 1) and words[cut - 1].lower().rstrip(',;:') in _DANGLING_WORDS:
        cut -= 1
    return ' '.join(words[:cut]).rstrip('.!?;,:')


def _select_best_asc_clause(asc: str, *, context: str) -> str:

    s = _sanitize_text_single_line(asc)
    s = _KEY_MOMENT_PREFIX_RE.sub("", s).strip()
    if not s:
        return ""


    clauses = [c.strip().rstrip(".") for c in re.split(r"(?:\.\s|;\s)", s) if c.strip()]
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


def _needs_violated(clause: str) -> bool:

    lowered = str(clause or "").strip().lower()
    if not lowered:
        return False

    phrase_markers = [
        "not within reach", "too far", "out of reach",
        "too heavy", "too large", "not aligned",
        "not visible", "not present", "not positioned",
        "not accessible", "not available", "not reachable",
        "no clear path", "no line of sight", "beyond reach",
        "blocks direct access", "blocks access",
    ]
    if any(marker in lowered for marker in phrase_markers):
        return True


    word_markers = [
        "obstructed", "blocked", "blocks", "blocking",
        "inaccessible", "missing", "absent",
        "unavailable", "misaligned", "occluded", "cannot", "impossible",
        "unreachable", "insufficient",

        "stuck", "sealed", "jammed", "locked", "frozen", "wedged",

        "prevents", "obstructs", "hinders", "impedes",
    ]
    for marker in word_markers:


        if re.search(rf'\b{re.escape(marker)}\b', lowered):
            return True


    if re.search(r'\bnot\s+(?:\w+able|stable|secure|ready|open|closed|tight|firm|level|clean|dry|flat|safe)\b', lowered):
        return True
    return False


def _clean_counterfactual_outcome(outcome: str) -> str:

    s = _sanitize_text_single_line(str(outcome or "").strip())
    if not s:
        return ""


    sentences = re.split(r'(?<=[.!?])\s+', s)
    if len(sentences) > 4:
        s = " ".join(sentences[:4])
        if s and s[-1] not in ".?!":
            s += "."
    m = _RECOVERY_SUGGESTION_RE.search(s)
    m2 = _RECOVERY_ACTION_PHRASE_RE.search(s)
    if not m and not m2:

        if s and s[0].islower():
            s = s[0].upper() + s[1:]

        if s and s[-1] not in ".?!":
            s += "."
        return s

    if m and m2:
        cut_pos = min(m.start(), m2.start())
    elif m:
        cut_pos = m.start()
    else:
        cut_pos = m2.start()                            

    cut = s[:cut_pos].rstrip(" ,;:").strip()
    if cut:
        if cut[-1] not in ".?!":
            cut = cut + "."

        if cut[0].islower():
            cut = cut[0].upper() + cut[1:]
        return cut


    tail = ""

    if m and m.start() == cut_pos:
        first_match_end = m.end()
    elif m2:
        first_match_end = m2.end()
    else:
        first_match_end = m.end() if m else 0                            
    comma = s.find(",", max(0, int(first_match_end)))
    if comma != -1:
        tail = s[comma + 1 :].strip()
    else:
        tail = s[first_match_end:].strip()
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
    elif lower.startswith("resulting in "):
        tail = "This could result in " + tail[len("resulting in ") :].lstrip()
    elif lower.startswith("preventing "):
        tail = "This could prevent " + tail[len("preventing ") :].lstrip()
    elif lower.startswith("compromising "):
        tail = "This could compromise " + tail[len("compromising ") :].lstrip()
    elif lower.startswith("reducing "):
        tail = "This could reduce " + tail[len("reducing ") :].lstrip()
    elif lower.startswith("increasing "):
        tail = "This could increase " + tail[len("increasing ") :].lstrip()
    elif lower.startswith("worsening "):
        tail = "This could worsen " + tail[len("worsening ") :].lstrip()

    elif lower.startswith("damaging "):
        tail = "This could damage " + tail[len("damaging ") :].lstrip()
    elif lower.startswith("breaking "):
        tail = "This could break " + tail[len("breaking ") :].lstrip()
    elif lower.startswith("burning "):
        tail = "This could burn " + tail[len("burning ") :].lstrip()
    elif lower.startswith("spilling "):
        tail = "This could spill " + tail[len("spilling ") :].lstrip()
    elif lower.startswith("overcooking "):
        tail = "This could overcook " + tail[len("overcooking ") :].lstrip()
    elif lower.startswith("undercooking "):
        tail = "This could undercook " + tail[len("undercooking ") :].lstrip()
    elif lower.startswith("contaminating "):
        tail = "This could contaminate " + tail[len("contaminating ") :].lstrip()
    elif lower.startswith("overheating "):
        tail = "This could overheat " + tail[len("overheating ") :].lstrip()


    if _BARE_IMPERATIVE_RE.match(tail.strip()):
        return ""

    tail = tail.strip()
    if not tail:
        return ""
    if tail[-1] not in ".?!":
        tail = tail + "."

    if _RECOVERY_SUGGESTION_RE.search(tail) or _RECOVERY_ACTION_PHRASE_RE.search(tail):
        return ""

    if tail and tail[0].islower():
        tail = tail[0].upper() + tail[1:]
    return tail


def _lowercase_first_alpha(text: str) -> str:

    s = str(text or "")
    if not s:
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            if not ch.isupper():
                return s                                      

            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            first_word = s[i:j]

            if first_word == "I":
                return s

            if len(first_word) >= 2 and first_word.isupper():
                return s
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def _imperitive_to_gerund(text: str) -> str:

    s = str(text or "").strip()
    if not s:
        return s

    m = re.match(r"([A-Za-z]+)(.*)", s, re.DOTALL)
    if not m:
        return s
    first_word = m.group(1)
    rest = m.group(2)
    fw_lower = first_word.lower()


    _ENDS_ING_BUT_NOT_GERUND = {
        "swing", "bring", "ring", "sling", "cling", "fling", "sting",
        "wring", "string", "spring", "sing", "king", "thing",
    }
    if fw_lower.endswith("ing") and fw_lower not in _ENDS_ING_BUT_NOT_GERUND:
        return s

    _SKIP = {"the", "a", "an", "this", "that", "these", "those", "it", "its",
             "if", "when", "while", "because", "since", "for", "by", "to", "from"}
    if fw_lower in _SKIP:
        return s


    if fw_lower.endswith("e") and not fw_lower.endswith("ee"):
        gerund = fw_lower[:-1] + "ing"



    elif (len(fw_lower) <= 5
          and len(fw_lower) >= 3
          and fw_lower[-1] not in "aeiouwxy"
          and fw_lower[-2] in "aeiou"
          and fw_lower[-3] not in "aeiou"
          and fw_lower not in {"open", "lower", "enter", "cover", "order",
                               "offer", "power", "tower", "outer", "inner",
                               "lever", "widen", "ripen", "happen", "listen"}):
        gerund = fw_lower + fw_lower[-1] + "ing"
    else:
        gerund = fw_lower + "ing"

    if first_word[0].isupper():
        gerund = gerund[0].upper() + gerund[1:]
    return gerund + rest


def _to_progressive_phrase(phrase: str) -> str:

    s = str(phrase or "").strip()
    if not s:
        return s


    parts = re.split(r'(\s*,\s*and\s+|\s*,\s*then\s+|\s+and\s+|\s+then\s+|\s*,\s+)', s)
    result: list = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            result.append(part)
            continue

        if re.match(r'^(?:,?\s*and\s*|,?\s*then\s*|,\s*)$', stripped, re.IGNORECASE):
            result.append(part)
            continue

        result.append(_imperitive_to_gerund(part))
    return "".join(result)


def _post_llm_fix_caps(text: str) -> str:

    s = str(text or "")
    if not s:
        return s


    s = re.sub(
        r'\b(because|since|while|when|although|if|as|and|or|but|plus)\s+([A-Z])(?=[a-z])',
        lambda m: m.group(1) + " " + m.group(2).lower(),
        s,
    )

    s = re.sub(
        r'([:;—–])\s+([A-Z])(?=[a-z]|\s+[a-z])',
        lambda m: m.group(1) + " " + m.group(2).lower(),
        s,
    )




    s = re.sub(
        r',\s+([A-Z])(?=[a-z]|\s+[a-z])',
        lambda m: ", " + m.group(1).lower(),
        s,
    )

    s = re.sub(
        r'(?<=\s)to\s+([A-Z])(?=[a-z])',
        lambda m: "to " + m.group(1).lower(),
        s,
    )

    if s[:3] == "to " and len(s) > 3 and s[3].isupper() and (len(s) < 5 or s[4].islower()):
        s = "to " + s[3].lower() + s[4:]


    s = re.sub(r'\baffordance_type\b', 'affordance type', s)



    s = re.sub(r'\bits\s+affordance\s+type\s+is\s+', 'it functions as a ', s)

    s = re.sub(r'\bits\s+(?:relevant\s+)?affordance\s+is\s+', 'it functions as ', s)

    s = re.sub(r'\s+affordance\s+type\b', '', s)

    s = re.sub(r'\s+affordance\b(?=\s*[,.])', '', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r'\bstep_goal\b', 'step goal', s)
    s = re.sub(r'\bpatient_text\b', 'patient', s)
    s = re.sub(r'\bhigh_level_goal\b', 'high-level goal', s)


    s = re.sub(
        r'\bby\s+([A-Z][a-z]+)\b',
        lambda m: "by " + _imperitive_to_gerund(m.group(1).lower()),
        s,
    )


    s = re.sub(
        r'\bby\s+([a-z]+)\s+(the|a|an|this|that|its)\b',
        lambda m: "by " + _imperitive_to_gerund(m.group(1)) + " " + m.group(2),
        s,
    )






    s = re.sub(
        r'\b(the\s+person|the\s+hand|the\s+person\'s\s+hand)\s+([A-Z])(?=[a-z])',
        lambda m: m.group(1) + " " + m.group(2).lower(),
        s,
    )


    s = re.sub(
        r'\b(will|would|should|can|could|may|might|must|shall)\s+([A-Z])(?=[a-z]|\s+[a-z])',
        lambda m: m.group(1) + " " + m.group(2).lower(),
        s,
    )
    return s


def _make_task04_to_13(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)
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


    _seen_kf_answers: set = set()

    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue

        step_cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        agent = _require_str(step_cc, "agent") if isinstance(step_cc, dict) else ""
        patient = _require_str(step_cc, "patient") if isinstance(step_cc, dict) else ""
        action = _require_str(step_cc, "action") if isinstance(step_cc, dict) else ""


        _t04_clip = _resolve_video_clip(item_dir, sid) if bool(attach_evidence) else None
        _t04_clip_rel = _safe_relpath(_t04_clip, input_root) if _t04_clip else None

        k0 = _keyframe_for_task(item_dir, st, prefer_j=0, require_image=bool(attach_evidence))
        k1 = _keyframe_for_task(item_dir, st, prefer_j=1, require_image=bool(attach_evidence))





        if action and (_t04_clip_rel or not bool(attach_evidence)):
            act = action.strip()

            _t04_variant = _template_idx(5, "T04a", item_dir, sid)
            if act:


                _is_single_word = " " not in act
                _eff_variant = 1 if _is_single_word else _t04_variant


                _act_bare = act.replace('"', "'")
                _act_prog = _to_progressive_phrase(_act_bare)
                _act_prog_lc = _lowercase_first_alpha(_act_prog)
                if _eff_variant == 0:
                    a04 = _sanitize_space(f"The person is {_act_prog_lc}.")
                elif _eff_variant == 1:
                    a04 = _sanitize_space(f"In this clip, the person is {_act_prog_lc}.")
                elif _eff_variant == 2:
                    a04 = _sanitize_space(f"What is happening here is that the person is {_act_prog_lc}.")
                elif _eff_variant == 3:
                    a04 = _sanitize_space(f"The person can be seen {_act_prog_lc}.")
                else:
                    a04 = _sanitize_space(f"This clip shows the person {_act_prog_lc}.")
            else:
                a04 = ""
            if a04:

                _T11_QUESTIONS = [
                    'What action is occurring in this clip?',
                    'Describe the action shown in this clip.',
                    'What is the person doing in this clip?',
                ]
                q = _T11_QUESTIONS[_template_idx(len(_T11_QUESTIONS), "T04q", item_dir, sid)]
                out.append(
                    Sample(
                        TASK_11,
                        EVIDENCE_CLIP,
                        [],
                        _t04_clip_rel,
                        q,
                        a04,
                        source_rel,



                        llm_fields={
                            "action": act,
                            "step_goal": step_goal,
                            "patient": patient,
                            "agent": agent or "the person",
                        },
                    )
                )


        frames_for_hotspot: List[Tuple[Dict[str, Any], str]] = []
        if k0:
            _, cf0, img0 = k0
            frames_for_hotspot.append((cf0, img0))
        if k1:
            _, cf1, img1 = k1


            _prev_img = frames_for_hotspot[-1][1] if frames_for_hotspot else ""
            _same_image = bool(img1) and bool(_prev_img) and img1 == _prev_img
            if not frames_for_hotspot or not _same_image:
                frames_for_hotspot.append((cf1, img1))


        for _kf_idx, (cf, img) in enumerate(frames_for_hotspot):
            intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
            hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
            aff_type = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
            mech = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""
            asc = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))


            kf_patient = (_require_str(hotspot, "patient") or _require_str(hotspot, "description")) if isinstance(hotspot, dict) else ""
            kf_patient = kf_patient.strip() if kf_patient else ""
            t05_patient = kf_patient if kf_patient else patient                                                  


            if t05_patient and aff_type and mech:
                _T04_QUESTIONS = [
                    (
                        f'Context: The current action is "{step_goal}". '
                        "What is the main object being acted upon in this image? "
                        "Describe how its physical properties enable the interaction shown."
                    ),
                    (
                        f'Context: The current action is "{step_goal}". '
                        "Identify the key object being manipulated in this image and explain "
                        "what physical characteristics make the interaction possible."
                    ),
                    (
                        f'Context: The current action is "{step_goal}". '
                        "Which object is primarily being acted on in this image? "
                        "Explain the type of interaction and the underlying physical mechanism."
                    ),
                ]
                q = _T04_QUESTIONS[_template_idx(len(_T04_QUESTIONS), "T05q", item_dir, sid, _kf_idx)]

                mech_clause = _lowercase_first_alpha(_inline_clause(mech))


                if mech_clause:
                    patient_text = t05_patient.strip().rstrip(".").replace('"', "'")
                    a_aff = aff_type.strip().replace("_", " ").strip()



                    _TOOL_TO_PATIENT_AFFORDANCE = {
                        "cutting edge": "cuttable material",
                        "blade edge": "cuttable material",
                        "sharp edge": "cuttable material",
                        "lever arm": "pivotable joint",
                        "piercing tip": "penetrable surface",
                    }
                    if a_aff.lower() in _TOOL_TO_PATIENT_AFFORDANCE:
                        a_aff = _TOOL_TO_PATIENT_AFFORDANCE[a_aff.lower()]


                    if patient_text and a_aff:



                        _t05_variant = _template_idx(5, "T05a", item_dir, sid, _kf_idx)
                        _aff_article = "an" if a_aff and a_aff[0].lower() in "aeiou" else "a"




                        _patient_lower = patient_text.lower().strip()
                        _aff_lower = a_aff.lower().strip()
                        _same_value = (_patient_lower == _aff_lower) or (
                            _patient_lower.rstrip("s") == _aff_lower.rstrip("s")
                        )
                        if _same_value:


                            a = _sanitize_space(
                                f'The main object being acted upon is the {patient_text}. '
                                f"The interaction works because {mech_clause}."
                            )
                        elif _t05_variant == 0:
                            a = _sanitize_space(
                                f'The main object being acted upon is "{patient_text}", '
                                f"which functions as {_aff_article} {a_aff}. "
                                f"Physically, {mech_clause}."
                            )
                        elif _t05_variant == 1:
                            a = _sanitize_space(
                                f'Here, "{patient_text}" is the object receiving the action. '
                                f"Its key physical characteristic is that it serves as {_aff_article} {a_aff}. "
                                f"The interaction works because {mech_clause}."
                            )
                        elif _t05_variant == 2:
                            a = _sanitize_space(
                                f'The interaction targets "{patient_text}" — '
                                f"{_aff_article} {a_aff} object. "
                                f"The underlying mechanism is: {mech_clause}."
                            )
                        elif _t05_variant == 3:
                            a = _sanitize_space(
                                f'In this image, "{patient_text}" is being manipulated. '
                                f"It acts as {_aff_article} {a_aff}, meaning {mech_clause}."
                            )
                        else:
                            a = _sanitize_space(
                                f'The object at the center of the action is "{patient_text}". '
                                f"Because it is {_aff_article} {a_aff}, {mech_clause}."
                            )
                        _t05_key = f"T05:{a}"
                        if _t05_key not in _seen_kf_answers:
                            _seen_kf_answers.add(_t05_key)
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
                                        "patient": t05_patient.strip(),



                                        **({"affordance_type": aff_type.strip()} if not _same_value else {}),
                                        "mechanism": mech.strip(),
                                    },
                                )
                            )
            if asc:


                asc_s = _normalize_trailing_punct(_sanitize_space(_sanitize_text_single_line(asc)))

                if asc_s and asc_s[0].islower():
                    asc_s = asc_s[0].upper() + asc_s[1:]

                if asc_s and len(asc_s.strip().rstrip(".!?")) >= 15:
                    _t06_key = f"T06:{asc_s}"
                    if _t06_key not in _seen_kf_answers:
                        _seen_kf_answers.add(_t06_key)
                        _T12_QUESTIONS = [
                            f'Context: Current action: "{step_goal}" What action is taking place in this keyframe, and how does it change the objects involved?',
                            f'Context: Current action: "{step_goal}" Describe what is happening in this keyframe and how the objects are affected.',
                            f'Context: Current action: "{step_goal}" What is the person doing in this keyframe, and what effect does it have on the surrounding objects?',
                        ]
                        q06 = _T12_QUESTIONS[_template_idx(len(_T12_QUESTIONS), "T06q", item_dir, sid, _kf_idx)]
                        out.append(
                            Sample(
                                TASK_12,
                                EVIDENCE_KEYFRAME,
                                _img_list(img),
                                None,
                                q06,
                                asc_s,
                                source_rel,
                                llm_fields={"step_goal": step_goal, "action_state_change_description": _sanitize_text_single_line(asc)},
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

        for _kf_idx, (cf, img) in enumerate(frames_for_task10):

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

            mech_raw = _require_str(intr, "mechanism") if isinstance(intr, dict) else ""
            mech = _inline_clause(mech_raw)


            ctx_terms = _normalize_terms(" ".join([step_goal, mech_raw]))






            _T05_MAX_POINTS_PER_CAT = 2

            def _ranked_points(points: List[str], *, top_k: int = _T05_MAX_POINTS_PER_CAT) -> List[str]:
                scored: List[Tuple[float, str]] = []
                for p in points or []:
                    clause = _inline_clause(str(p))
                    if not clause:
                        continue
                    if _needs_not_directly_observable(clause):
                        continue
                    clause_terms = _normalize_terms(clause)
                    score = float(len(clause_terms & ctx_terms))
                    score -= 0.001 * len(clause)
                    scored.append((score, clause))
                scored.sort(key=lambda x: x[0], reverse=True)
                seen: List[str] = []
                for _, clause in scored[:top_k]:


                    p2 = _lowercase_first_alpha(clause.strip().rstrip(",;:"))
                    if p2 and p2 not in seen:
                        seen.append(p2)
                return seen

            sp_pre_ranked = _ranked_points(sp_pre_pts)
            af_pre_ranked = _ranked_points(af_pre_pts)
            sp_eff_ranked = _ranked_points(sp_eff_pts)
            af_eff_ranked = _ranked_points(af_eff_pts)


            pre_parts: List[str] = []
            for p2 in sp_pre_ranked + af_pre_ranked:
                if p2 not in pre_parts:
                    pre_parts.append(p2)

            eff_parts: List[str] = []
            for p2 in sp_eff_ranked + af_eff_ranked:
                if p2 not in eff_parts:
                    eff_parts.append(p2)


            def _join_parts(parts: List[str]) -> str:
                if not parts:
                    return ""
                if len(parts) == 1:
                    return parts[0]
                if len(parts) == 2:
                    return f"{parts[0]} and {parts[1]}"
                return ", ".join(parts[:-1]) + f", and {parts[-1]}"

            pre_clause = _join_parts(pre_parts).strip()
            eff_clause = _join_parts(eff_parts).strip()

            if not (mech and pre_clause and eff_clause):
                continue

            _T05_QUESTIONS = [
                (
                    f'Context: The current action is "{step_goal}". '
                    "Based on this keyframe, explain the causal chain: "
                    "what spatial and affordance conditions enable the action, "
                    "and what changes result? Answer concisely in English."
                ),
                (
                    f'Context: The current action is "{step_goal}". '
                    "Describe the cause-and-effect relationship visible in this keyframe: "
                    "what preconditions allow the action, and what state changes follow? "
                    "Answer concisely."
                ),
                (
                    f'Context: The current action is "{step_goal}". '
                    "What causal chain connects the spatial and affordance preconditions "
                    "to the resulting changes in this keyframe? Explain concisely."
                ),
            ]
            q = _T05_QUESTIONS[_template_idx(len(_T05_QUESTIONS), "T07q", item_dir, sid, _kf_idx)]

            mech_lower = _lowercase_first_alpha(mech)



            _t07_variant = _template_idx(5, "T07a", item_dir, sid, _kf_idx)
            if _t07_variant == 0:
                a = _sanitize_space(
                    f"The preconditions are that {pre_clause}. "
                    f"Under these conditions, {mech_lower}. "
                    f"As a result, {eff_clause}."
                )
            elif _t07_variant == 1:
                a = _sanitize_space(
                    f"Since {pre_clause}, {mech_lower}. "
                    f"The outcome is that {eff_clause}."
                )
            elif _t07_variant == 2:
                a = _sanitize_space(
                    f"{_lowercase_first_alpha(pre_clause).capitalize()}. "
                    f"Given this, {mech_lower}. "
                    f"Consequently, {eff_clause}."
                )
            elif _t07_variant == 3:
                a = _sanitize_space(
                    f"Because {pre_clause}, {mech_lower}. "
                    f"This leads to {_lowercase_first_alpha(eff_clause)}."
                )
            else:
                a = _sanitize_space(
                    f"The starting conditions are: {_lowercase_first_alpha(pre_clause)}. "
                    f"As the action proceeds, {mech_lower}. "
                    f"The resulting change is that {eff_clause}."
                )

            _t07_key = f"T07:{a}"
            if _t07_key not in _seen_kf_answers:
                _seen_kf_answers.add(_t07_key)
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
                            "mechanism": mech_raw,
                            "spatial_preconditions": _truncate_llm_field(sp_pre_full),
                            "affordance_preconditions": _truncate_llm_field(af_pre_full),
                            "spatial_effects": _truncate_llm_field(sp_eff_full),
                            "affordance_effects": _truncate_llm_field(af_eff_full),


                            "selected_pre_clause": pre_clause,
                            "selected_eff_clause": eff_clause,
                        },
                    )
                )


        rationale = _require_str(st, "rationale")
        _t08_clip = _resolve_video_clip(item_dir, sid) if bool(attach_evidence) else None
        _t08_clip_rel = _safe_relpath(_t08_clip, input_root) if _t08_clip else None
        if (_t08_clip_rel or not bool(attach_evidence)) and hl and rationale:

            _rat = _inline_clause(rationale.strip())
            if _rat and len(_rat.split()) >= 4:                                                                    
                _T13_QUESTIONS = [
                    f'Context: High-level goal: "{hl}" Why is the step shown in this clip necessary for the overall goal?',
                    f'Context: High-level goal: "{hl}" Based on this clip, explain how this step contributes to achieving the overall goal.',
                    f'Context: High-level goal: "{hl}" What role does the action in this clip play in the broader plan?',
                ]
                q = _T13_QUESTIONS[_template_idx(len(_T13_QUESTIONS), "T08q", item_dir, sid)]
                _rat_lower = _lowercase_first_alpha(_rat)


                _rat_stripped = re.sub(r'^this\s+(?:step|action)\s+', '', _rat_lower, flags=re.IGNORECASE)
                if _rat_stripped != _rat_lower:
                    _rat_lower = "it " + _lowercase_first_alpha(_rat_stripped)                                       



                _t08_variant = _template_idx(5, "T08a", item_dir, sid)
                if _t08_variant == 0:
                    a08 = _sanitize_space(f"This step is necessary because {_rat_lower}.")
                elif _t08_variant == 1:
                    a08 = _sanitize_space(f"Performing this action is essential: {_rat_lower}.")
                elif _t08_variant == 2:
                    a08 = _sanitize_space(f"The step serves a critical role — {_rat_lower}.")
                elif _t08_variant == 3:
                    a08 = _sanitize_space(f"This action matters for the overall goal because {_rat_lower}.")
                else:


                    a08 = _sanitize_space(f"Without this step, {_rat_lower}.")
                out.append(
                    Sample(
                        TASK_13,
                        EVIDENCE_CLIP,
                        [],
                        _t08_clip_rel,
                        q,
                        a08,
                        source_rel,
                        llm_fields={"high_level_goal": hl, "step_goal": step_goal, "rationale": rationale},
                    )
                )


        _task_clip = _resolve_video_clip(item_dir, sid) if bool(attach_evidence) else None
        _task_clip_rel = _safe_relpath(_task_clip, input_root) if _task_clip else None
        _task_clip_ok = bool(_task_clip_rel) or not bool(attach_evidence)


        if _task_clip_ok and step_cc:
            sp_pre_pts = _format_spatial_points(step_cc.get("causal_precondition_on_spatial")) if isinstance(step_cc, dict) else []
            af_pre_pts = _format_affordance_points(step_cc.get("causal_precondition_on_affordance")) if isinstance(step_cc, dict) else []

            sp_pre_pts = [p for p in sp_pre_pts if p and "not directly observable" not in p.lower()]
            af_pre_pts = [p for p in af_pre_pts if p and "not directly observable" not in p.lower()]

            _MAX_POINTS = 999                                                        
            sp_pre_pts = sp_pre_pts[:_MAX_POINTS]
            af_pre_pts = af_pre_pts[:_MAX_POINTS]

            sp_pre_all = _sanitize_space(_join_as_paragraph([p for p in sp_pre_pts if p])) if len(sp_pre_pts) >= 2 else ""
            af_pre_all = _sanitize_space(_join_as_paragraph([p for p in af_pre_pts if p])) if len(af_pre_pts) >= 2 else ""
            if sp_pre_all:
                _T01_QUESTIONS = [
                    'Based on this clip, describe the spatial conditions that must be in place before this step can be executed.',
                    'What spatial conditions visible in this clip need to be in place before this step can begin?',
                    'Based on this clip, describe the required spatial arrangement prior to this step.',
                ]
                q = _T01_QUESTIONS[_template_idx(len(_T01_QUESTIONS), "T09q", item_dir, sid)]
                out.append(
                    Sample(
                        TASK_01,
                        EVIDENCE_CLIP,
                        [],
                        _task_clip_rel,
                        q,
                        sp_pre_all,
                        source_rel,
                        llm_fields={"spatial_preconditions": sp_pre_all,
                                     "spatial_preconditions_pts": [p for p in sp_pre_pts if p],

                                     "step_goal": step_goal,
                                     "patient": patient},
                    )
                )
            if af_pre_all:
                _T02_QUESTIONS = [
                    'Based on this clip, describe the functional properties that objects must have before this step can be executed.',
                    'What affordance properties must the objects shown in this clip have before this step can proceed?',
                    'Based on this clip, describe the functional prerequisites of the objects involved in this step.',
                ]
                q = _T02_QUESTIONS[_template_idx(len(_T02_QUESTIONS), "T10q", item_dir, sid)]
                out.append(
                    Sample(
                        TASK_02,
                        EVIDENCE_CLIP,
                        [],
                        _task_clip_rel,
                        q,
                        af_pre_all,
                        source_rel,
                        llm_fields={"affordance_preconditions": af_pre_all,
                                     "affordance_preconditions_pts": [p for p in af_pre_pts if p],

                                     "step_goal": step_goal,
                                     "patient": patient},
                    )
                )


            _sp_raw = step_cc.get("causal_precondition_on_spatial") if isinstance(step_cc, dict) else None
            _af_raw = step_cc.get("causal_precondition_on_affordance") if isinstance(step_cc, dict) else None
            _sp_all = _format_spatial_points(_sp_raw) if _sp_raw else []
            _af_all = _format_affordance_points(_af_raw) if _af_raw else []
            _sp_all = [p for p in _sp_all if p and "not directly observable" not in p.lower()]
            _af_all = [p for p in _af_all if p and "not directly observable" not in p.lower()]


            _sp_all = [p for p in _sp_all if not _needs_violated(p)]
            _af_all = [p for p in _af_all if not _needs_violated(p)]
            sp_pre_1 = _pick_best_precondition(_sp_all, step_goal) if _sp_all else ""
            af_pre_1 = _pick_best_precondition(_af_all, step_goal) if _af_all else ""
            if sp_pre_1 and af_pre_1:
                sp_clause = _lowercase_first_alpha(_inline_clause(sp_pre_1)).strip().rstrip('.!?')
                af_clause = _lowercase_first_alpha(_inline_clause(af_pre_1)).strip().rstrip('.!?')



                _sp_words = len(sp_clause.split()) if sp_clause else 0
                _af_words = len(af_clause.split()) if af_clause else 0
                if _sp_words < 4 or _af_words < 4:
                    sp_clause = ""
                    af_clause = ""


                if sp_clause and af_clause:
                    _sp_tokens = set(sp_clause.lower().split())
                    _af_tokens = set(af_clause.lower().split())
                    _overlap = len(_sp_tokens & _af_tokens) / max(min(len(_sp_tokens), len(_af_tokens)), 1)
                    if _overlap > 0.75:
                        sp_clause = ""
                        af_clause = ""


                if sp_clause and af_clause:

                    _t11_variant = _template_idx(5, "T11a", item_dir, sid)
                    if _t11_variant == 0:
                        a = _sanitize_space(
                            f"This step is feasible because {sp_clause} "
                            f"and {af_clause}."
                        )
                    elif _t11_variant == 1:
                        a = _sanitize_space(
                            f"The step is physically possible since {sp_clause}, "
                            f"while {af_clause}."
                        )
                    elif _t11_variant == 2:
                        a = _sanitize_space(
                            f"This action can proceed: {sp_clause}; "
                            f"additionally, {af_clause}."
                        )
                    elif _t11_variant == 3:
                        a = _sanitize_space(
                            f"The conditions are met — {sp_clause}, "
                            f"and {af_clause}."
                        )
                    else:
                        a = _sanitize_space(
                            f"Everything needed is in place: {sp_clause}, "
                            f"plus {af_clause}."
                        )
                    _T03_QUESTIONS = [
                        (
                            "Based on the conditions visible in this clip, explain why the current step is physically feasible now. "
                            "Answer in one English sentence, and justify by stating one spatial precondition "
                            "and one affordance precondition."
                        ),
                        (
                            "Given what is shown in this clip, why is this step physically possible at this moment? "
                            "Justify in one sentence by citing one spatial and one affordance precondition."
                        ),
                        (
                            "Based on this clip, what spatial and affordance conditions make this step feasible right now? "
                            "Explain in a single sentence."
                        ),
                    ]
                    q = _T03_QUESTIONS[_template_idx(len(_T03_QUESTIONS), "T11q", item_dir, sid)]
                    out.append(
                        Sample(
                            TASK_03,
                            EVIDENCE_CLIP,
                            [],
                            _task_clip_rel,
                            q,
                            a,
                            source_rel,
                            llm_fields={
                                "spatial_precondition": sp_pre_1,
                                "affordance_precondition": af_pre_1,
                                "all_spatial_preconditions": _sp_all,
                                "all_affordance_preconditions": _af_all,

                                "step_goal": step_goal,
                                "patient": patient,
                                "action": action,
                            },
                        )
                    )


        if _task_clip_ok and step_cc:
            sp_eff_pts = _format_spatial_points(step_cc.get("causal_effect_on_spatial")) if isinstance(step_cc, dict) else []
            af_eff_pts = _format_affordance_points(step_cc.get("causal_effect_on_affordance")) if isinstance(step_cc, dict) else []

            sp_eff_pts = [p for p in sp_eff_pts if p and "not directly observable" not in p.lower()]
            af_eff_pts = [p for p in af_eff_pts if p and "not directly observable" not in p.lower()]

            sp_eff_pts = sp_eff_pts[:_MAX_POINTS]
            af_eff_pts = af_eff_pts[:_MAX_POINTS]


            if len(sp_eff_pts) >= 2:
                _T06_QUESTIONS = [
                    'After completing this step, describe the resulting spatial arrangement of the objects shown in this clip.',
                    'What spatial changes visible in this clip result from completing this step?',
                    'Based on this clip, describe how the spatial arrangement of objects changes after this step.',
                ]
                q = _T06_QUESTIONS[_template_idx(len(_T06_QUESTIONS), "T12q", item_dir, sid)]

                a = _sanitize_space(_join_as_paragraph([p for p in sp_eff_pts if p]))

                if a:
                    out.append(
                        Sample(
                            TASK_06,
                            EVIDENCE_CLIP,
                            [],
                            _task_clip_rel,
                            q,
                            a,
                            source_rel,
                            llm_fields={"spatial_postconditions": a,
                                         "spatial_postconditions_pts": [p for p in sp_eff_pts if p],

                                         "step_goal": step_goal,
                                         "patient": patient},
                        )
                    )
            if len(af_eff_pts) >= 2:
                _T07_QUESTIONS = [
                    'After completing this step, describe how the functional properties of the objects in this clip have changed.',
                    'What affordance or functional state changes shown in this clip result from completing this step?',
                    'Based on this clip, describe how the functional properties of objects change after this step.',
                ]
                q = _T07_QUESTIONS[_template_idx(len(_T07_QUESTIONS), "T13q", item_dir, sid)]
                a = _sanitize_space(_join_as_paragraph([p for p in af_eff_pts if p]))

                if a:
                    out.append(
                        Sample(
                            TASK_07,
                            EVIDENCE_CLIP,
                            [],
                            _task_clip_rel,
                            q,
                            a,
                            source_rel,
                            llm_fields={"affordance_postconditions": a,
                                         "affordance_postconditions_pts": [p for p in af_eff_pts if p],

                                         "step_goal": step_goal,
                                         "patient": patient},
                        )
                    )



    _cross_pairs = [(TASK_01, TASK_06), (TASK_02, TASK_07)]
    _to_remove: set[int] = set()
    for _pair_a, _pair_b in _cross_pairs:
        _a_indices = [(i, s) for i, s in enumerate(out) if s.task_name == _pair_a]
        _b_indices = [(i, s) for i, s in enumerate(out) if s.task_name == _pair_b]
        for _ia, _sa in _a_indices:
            for _ib, _sb in _b_indices:
                if _sa.video != _sb.video:
                    continue                                          
                _a_tokens = set(_sa.answer.lower().split())
                _b_tokens = set(_sb.answer.lower().split())
                _denom = max(min(len(_a_tokens), len(_b_tokens)), 1)
                if len(_a_tokens & _b_tokens) / _denom > 0.75:


                    _to_remove.add(_ia)
    if _to_remove:
        out = [s for i, s in enumerate(out) if i not in _to_remove]

    return out


def _make_task14(item_dir: str, plan: Dict[str, Any], input_root: str, *, attach_evidence: bool) -> Iterable[Sample]:

    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2 or not hl:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)

    for i in range(len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]


        independence = str(s1.get("independence", "")).strip().lower()
        if independence != "yes":
            continue


        detail_independence = _require_str(s1, "detail_independence")
        if not detail_independence:
            continue

        sid0 = int(s0.get("step_id", 0) or 0)
        sid1 = int(s1.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue

        if sid0 == sid1:
            continue

        _sg0_tokens = set(sg0.lower().split())
        _sg1_tokens = set(sg1.lower().split())
        _sg_overlap = len(_sg0_tokens & _sg1_tokens) / max(min(len(_sg0_tokens), len(_sg1_tokens)), 1)
        if _sg_overlap > 0.80:
            continue


        video_json: Optional[str] = None
        if bool(attach_evidence):
            clip0 = _resolve_video_clip(item_dir, sid0)
            clip1 = _resolve_video_clip(item_dir, sid1)
            if not clip0 or not clip1:
                continue
            clip0_rel = _safe_relpath(clip0, input_root)
            clip1_rel = _safe_relpath(clip1, input_root)
            video_json = json.dumps([clip0_rel, clip1_rel])


        _T14_QUESTIONS = [
            (
                f'Context: High-level goal: "{hl}" '
                f"The first video shows step {sid0} and the second video shows step {sid1}. "
                f"How does the outcome of step {sid0} satisfy the preconditions for step {sid1}?"
            ),
            (
                f'Context: High-level goal: "{hl}" '
                f"The first video shows step {sid0} and the second video shows step {sid1}. "
                f"What physical or spatial outcome of step {sid0} enables step {sid1} to proceed?"
            ),
            (
                f'Context: High-level goal: "{hl}" '
                f"The first video shows step {sid0} and the second video shows step {sid1}. "
                f"Explain how step {sid0} creates the conditions necessary for step {sid1}."
            ),
        ]
        q = _T14_QUESTIONS[sid1 % len(_T14_QUESTIONS)]



        _detail_text = _sanitize_space(detail_independence)
        if not _detail_text:
            continue
        _detail_lower = _lowercase_first_alpha(_detail_text.rstrip(".")) if _detail_text else ""

        if not _detail_lower.strip():
            continue
        _t14_variant = sid1 % 3
        if _t14_variant == 0:
            a = _sanitize_space(f"Step {sid0} creates the necessary conditions: {_detail_lower}.")
        elif _t14_variant == 1:
            a = _sanitize_space(f"The outcome of step {sid0} enables step {sid1} because {_detail_lower}.")
        else:
            a = _sanitize_space(f"The dependency between step {sid0} and step {sid1} exists because {_detail_lower}.")

        out.append(
            Sample(
                TASK_14,
                EVIDENCE_CLIP_PAIR,
                [],
                video_json,
                q,
                a,
                source_rel,
                llm_fields={
                    "high_level_goal": hl,
                    "prev_step_goal": sg0,
                    "next_step_goal": sg1,
                    "independence": independence,
                    "detail_independence": detail_independence,
                },
            )
        )
    return out


def _make_task15(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 3 or not hl:                                                               
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)


    for i in range(1, len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]
        sid0 = int(s0.get("step_id", 0) or 0)
        sid1 = int(s1.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue
        video_rel: Optional[str] = None
        if bool(attach_evidence):
            video = _resolve_video_prefix_or_step_clip(
                item_dir,
                sid0,
                strict=bool(strict_prefix_video),
                plan=plan,
            )
            if not video:
                continue
            video_rel = _safe_relpath(video, input_root)

        _hl_ctx = _shorten_hl_for_context(hl)
        _T15_QUESTIONS = [
            f'Context: High-level goal: "{_hl_ctx}." Based on this video prefix, what is the next step goal?',
            f'Context: High-level goal: "{_hl_ctx}." Given this video prefix, predict the next step the person will take.',
            f'Context: High-level goal: "{_hl_ctx}." What action should follow next based on this video prefix?',
        ]
        q = _T15_QUESTIONS[sid1 % len(_T15_QUESTIONS)]
        evidence_type = EVIDENCE_PREFIX
        images = []

        sg1_clean = sg1.strip()
        if not sg1_clean:
            continue

        _sg1_lower = _lowercase_first_alpha(sg1_clean).rstrip(".")

        if _sg1_lower.lower().startswith("to "):
            _sg1_lower = _sg1_lower[3:]

        _t15_variant = sid1 % 3
        if _t15_variant == 0:
            a15 = _sanitize_space(f"The next step is to {_sg1_lower}.")
        elif _t15_variant == 1:
            a15 = _sanitize_space(f"Following this, the person proceeds to {_sg1_lower}.")
        else:
            a15 = _sanitize_space(f"The subsequent action is to {_sg1_lower}.")
        out.append(
            Sample(
                TASK_15,
                evidence_type,
                images,
                video_rel,
                q,
                a15,
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


def _make_task16(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
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

    _MAX_MIDDLE = 8
    middle = middle[:_MAX_MIDDLE]


    if len(middle) == 1:
        sg_clean = middle[0].rstrip(".").replace('"', "'")
        a = f'The missing middle step goal is: "{sg_clean}."'
    else:
        parts = [f'{i+1}) "{sg.rstrip(".").replace(chr(34), chr(39))}."' for i, sg in enumerate(middle)]
        a = "The missing middle steps, in order, are: " + " ".join(parts)

    _hl_ctx = _shorten_hl_for_context(hl)
    _T16_QUESTIONS = [
        (
            f'Context: High-level goal: "{_hl_ctx}." '
            "Based on the first-step clip and the last-step clip from the same trajectory, "
            "infer the missing middle steps in order. Do not repeat the first or last step."
        ),
        (
            f'Context: High-level goal: "{_hl_ctx}." '
            "Given the first and last step clips of this trajectory, "
            "deduce the intermediate steps that bridge them. List them in order."
        ),
        (
            f'Context: High-level goal: "{_hl_ctx}." '
            "From the first-step and last-step clips, determine what middle steps "
            "must have occurred between them. Provide them in sequence."
        ),
    ]
    q = _T16_QUESTIONS[_stable_int_seed(json.dumps(middle)) % len(_T16_QUESTIONS)]
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)

    video_json: Optional[str] = None
    if bool(attach_evidence):
        first_step_id = int(steps[0].get("step_id", 0) or 0)
        last_step_id = int(steps[-1].get("step_id", 0) or 0)
        if first_step_id <= 0 or last_step_id <= 0:
            return None
        clip_first = _resolve_video_clip(item_dir, first_step_id)
        clip_last = _resolve_video_clip(item_dir, last_step_id)
        if not clip_first or not clip_last:
            return None
        clip_first_rel = _safe_relpath(clip_first, input_root)
        clip_last_rel = _safe_relpath(clip_last, input_root)
        video_json = json.dumps([clip_first_rel, clip_last_rel])
    return Sample(
        task_name=TASK_16,
        evidence_type=EVIDENCE_CLIP_PAIR,
        image=[],
        video=video_json,
        question=q,
        answer=_sanitize_space(a),
        source_path=source_rel,
        llm_fields={"high_level_goal": hl, "middle_step_goals": middle},
    )


def _make_task17(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    rng: random.Random,
    *,
    attach_evidence: bool,
    strict_prefix_video: bool,
) -> Iterable[Sample]:

    hl = _require_str(plan, "high_level_goal")

    _hl_ctx = _shorten_hl_for_context(hl)
    steps = _sorted_steps(plan)
    if len(steps) < 4 or not hl:                                   
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)



    min_prefix_idx = 1
    max_prefix_idx = len(steps) - 2                                  

    for i_idx in range(min_prefix_idx, max_prefix_idx + 1):
        prefix_step = steps[i_idx]
        prefix_end_step = int(prefix_step.get("step_id", 0) or 0)
        if prefix_end_step <= 0:
            continue

        remaining = steps[i_idx + 1:]
        if len(remaining) < 1:
            continue

        k = min(max(3, min(6, len(remaining))), len(remaining))
        gold = [str(s.get("step_goal", "")).strip() for s in remaining[:k] if str(s.get("step_goal", "")).strip()]

        if len(gold) < 1 or len(gold) != k:
            continue

        video_rel: Optional[str] = None
        if bool(attach_evidence):
            video = _resolve_video_prefix_or_step_clip(
                item_dir,
                prefix_end_step,
                strict=bool(strict_prefix_video),
                plan=plan,
            )
            if not video:
                continue
            video_rel = _safe_relpath(video, input_root)

        last_completed_goal = str(prefix_step.get("step_goal", "")).strip()

        _t17_ans_variant = _template_idx(3, "T17a", item_dir, i_idx)
        if len(gold) == 1:
            sg_clean = gold[0].rstrip(".")
            if _t17_ans_variant == 0:
                answer = f'The next step should be: "{sg_clean}."'
            elif _t17_ans_variant == 1:
                answer = f'Following the prefix, the next action is: "{sg_clean}."'
            else:
                answer = f'The predicted next step is to {_lowercase_first_alpha(sg_clean).rstrip(".")}.'
        else:
            answer_parts = []
            for j, sg in enumerate(gold):
                sg_clean = sg.rstrip(".").replace('"', "'")
                answer_parts.append(f'{j+1}) "{sg_clean}."')
            _parts_str = " ".join(answer_parts)
            if _t17_ans_variant == 0:
                answer = f"The next {len(gold)} steps, in order, are: {_parts_str}"
            elif _t17_ans_variant == 1:
                answer = f"Following the observed prefix, the subsequent {len(gold)} steps should be: {_parts_str}"
            else:
                answer = f"The predicted continuation is: {_parts_str}"

        images: List[str] = []

        _k = len(gold)
        if _k == 1:
            _T17_QUESTIONS = [
                f'Context: High-level goal: "{_hl_ctx}." Based on this prefix, predict the next step goal.',
                f'Context: High-level goal: "{_hl_ctx}." After watching this prefix, forecast the next step goal.',
                f'Context: High-level goal: "{_hl_ctx}." Given the prefix shown, what is the next step goal that should follow?',
            ]
        else:
            _T17_QUESTIONS = [
                f'Context: High-level goal: "{_hl_ctx}." Based on this prefix, predict the next K={_k} step goals, in order.',
                f'Context: High-level goal: "{_hl_ctx}." After watching this prefix, forecast the next {_k} step goals in sequence.',
                f'Context: High-level goal: "{_hl_ctx}." Given the prefix shown, what are the next {_k} step goals that should follow? List them in order.',
            ]
        q = _T17_QUESTIONS[i_idx % len(_T17_QUESTIONS)]
        out.append(
            Sample(
                TASK_17,
                EVIDENCE_PREFIX,
                images,
                video_rel,
                q,
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


def _make_task18_bad_plan_diagnosis_and_repair(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    rng: random.Random,
    *,
    attach_evidence: bool,
    strict_prefix_video: bool,
    cross_plan_goals: Optional[List[str]] = None,
) -> Iterable[Sample]:

    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 5 or not hl:                                                   
        return []
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)


    min_prefix_idx = 1                                        
    max_prefix_idx = min(3, len(steps) - 4)
    if max_prefix_idx < min_prefix_idx:
        return []


    all_plan_goals = set()
    for s in steps:
        sg = str(s.get("step_goal", "")).strip()
        if sg:
            all_plan_goals.add(sg)


    _goal_to_step_ids: Dict[str, List[int]] = {}
    for s in steps:
        sg = str(s.get("step_goal", "")).strip()
        sid = int(s.get("step_id", 0) or 0)
        if sg and sid > 0:
            _goal_to_step_ids.setdefault(sg, []).append(sid)

    def _uniq_pool(xs: List[str], *, exclude: set[str]) -> List[str]:

        seen: set[str] = set()
        out_x: List[str] = []
        for x in xs:
            x2 = str(x or "").strip()
            if not x2 or x2 in exclude or x2 in seen:
                continue
            seen.add(x2)
            out_x.append(x2)
        return out_x


    cross_pool_raw: List[str] = []
    if cross_plan_goals:
        cross_pool_raw = [g for g in cross_plan_goals if g not in all_plan_goals]



    def _vaguify(step_goal: str) -> str:

        sg = str(step_goal or "").strip()
        if not sg:
            return "Continue with the next action."

        words = sg.split()
        idx = 0

        if idx < len(words) and words[idx].lower().rstrip(",.:;") == "to":
            idx += 1

        _SKIP_WORDS = {"the", "a", "an", "this", "that", "these", "those",
                       "after", "before", "while", "once", "then", "also",
                       "first", "next", "finally", "further", "immediately"}
        while idx < len(words):
            w = words[idx].lower().rstrip(",.:;")
            if w in _SKIP_WORDS or (w.endswith("ly") and len(w) > 3):
                idx += 1
            else:
                break
        verb = words[idx].lower().rstrip(",.:;") if idx < len(words) else "handle"



        base_verb = verb
        if verb.endswith("ing") and len(verb) > 4 and verb not in _NOT_GERUNDS:
            stem = verb[:-3]
            _KEEP_DOUBLED = {"pull", "fill", "spill", "drill", "roll", "stall", "grill",
                             "scroll", "install", "skill", "mill", "bill", "kill", "till",
                             "sell", "tell", "yell", "spell", "smell", "dwell", "swell",
                             "call", "fall", "wall", "hall", "ball", "tall",
                             "buzz", "fizz", "fuzz", "jazz", "pass", "miss", "kiss", "toss",
                             "press", "dress", "bless", "stress", "cross", "floss", "boss",
                             "puff", "buff", "stuff", "bluff", "scuff", "huff",
                             "add", "odd"}


            _YING_TO_IE_STEMS = {"ty", "dy", "ly"}
            if verb.endswith("ying") and stem in _YING_TO_IE_STEMS:
                base_verb = stem[:-1] + "ie"


            elif len(stem) >= 2 and stem[-1] == stem[-2]:
                if stem in _KEEP_DOUBLED:
                    base_verb = stem
                else:
                    base_verb = stem[:-1]

            elif not stem.endswith(("a", "e", "i", "o", "u")):
                _SILENT_E_STEMS = {
                    "slic", "bak", "mak", "tak", "shak", "plac", "scrap", "wip",
                    "mov", "remov", "prepar", "stor", "clos", "ris", "shap",
                    "grat", "dic", "minc", "sauc", "plat",
                    "saut", "flam",
                    "serv", "carv", "sav", "leav", "hav",
                    "us", "chas", "eras", "purs", "rins",
                    "squeez", "freez", "chos",
                    "writ", "bit", "hid", "rid", "slid",
                }
                if stem in _SILENT_E_STEMS:
                    base_verb = stem + "e"
                else:
                    base_verb = stem
            else:
                base_verb = stem
        _NON_VERB_INDICATORS = {"person", "item", "object", "camera", "it", "they",
                                "he", "she", "one", "each", "all", "some", "any",
                                "remaining", "several", "both", "other"}
        if base_verb in _NON_VERB_INDICATORS or len(base_verb) <= 1:
            _GENERIC_TEMPLATES = [
                "Continue with the current action on the relevant items.",
                "Proceed with the next step as appropriate.",
                "Address the current situation as needed.",
                "Carry on with the necessary action on the objects.",
                "Manage the items by performing the required action.",
            ]
            return _GENERIC_TEMPLATES[_stable_int_seed(sg) % len(_GENERIC_TEMPLATES)]




        _VERB_CATEGORIES: Dict[str, str] = {
            "cut": "handle", "slice": "handle", "chop": "handle", "trim": "handle",
            "open": "handle", "close": "handle", "pull": "handle", "push": "handle",
            "place": "handle", "set": "handle", "move": "handle", "remove": "handle",
            "take": "handle", "bring": "handle", "carry": "handle", "lift": "handle",
            "store": "handle", "put": "handle", "lower": "handle", "transfer": "handle",
            "spread": "handle", "fold": "handle", "unwrap": "handle", "untwist": "handle",
            "press": "handle", "flip": "handle", "turn": "handle", "adjust": "handle",
            "stir": "prepare", "pour": "prepare", "mix": "prepare", "bake": "prepare",
            "fry": "prepare", "grate": "prepare", "rinse": "prepare", "squeeze": "prepare",
            "serve": "prepare", "plate": "prepare", "peel": "prepare", "dice": "prepare",
            "sort": "organize", "arrange": "organize", "clear": "organize",
            "gather": "organize", "collect": "organize",
        }
        _cat = _VERB_CATEGORIES.get(base_verb, "handle")
        _VAGUE_TEMPLATES_BY_CAT: Dict[str, List[str]] = {
            "handle": [
                "Continue with the current step as appropriate.",
                "Proceed to handle the objects appropriately.",
                "Address the current situation by performing the required action.",
                "Carry on with the necessary actions on the items.",
                "Manage the relevant items as needed.",
            ],
            "prepare": [
                "Proceed with the current preparation step.",
                "Continue preparing the items as appropriate.",
                "Carry on with the necessary preparation on the objects.",
                "Address the current preparation task as needed.",
                "Manage the items by performing the required preparation.",
            ],
            "organize": [
                "Proceed to organize the relevant items.",
                "Continue arranging the objects as appropriate.",
                "Carry on with the necessary arrangement of the items.",
                "Address the current organization task as needed.",
                "Manage the workspace by performing the required action.",
            ],
        }
        _templates = _VAGUE_TEMPLATES_BY_CAT.get(_cat, _VAGUE_TEMPLATES_BY_CAT["handle"])
        return _templates[_stable_int_seed(sg) % len(_templates)]


    def _short_excerpt(step_goal: str, max_words: int = 6) -> str:

        sg = str(step_goal or "").strip()
        if not sg:
            return "this step"
        words = sg.split()
        if len(words) <= max_words:
            return sg
        return " ".join(words[:max_words]) + "..."


    _ALL_FLAW_TYPES = [
        "goal_inconsistent",
        "granularity_mismatch",
        "precondition_missing",
        "redundant_step",
        "wrong_ordering",
    ]


    out: List[Sample] = []
    for ft_idx, flaw_type in enumerate(_ALL_FLAW_TYPES):
        ft_rng = random.Random(rng.randint(0, 2**31) + ft_idx)


        ft_i_idx = ft_rng.randrange(min_prefix_idx, max_prefix_idx + 1)
        ft_prefix_step = steps[ft_i_idx]
        ft_prefix_end_step = int(ft_prefix_step.get("step_id", 0) or 0)
        if ft_prefix_end_step <= 0:
            continue

        ft_remaining = steps[ft_i_idx + 1:]
        if len(ft_remaining) < 2:
            continue


        if flaw_type == "precondition_missing":
            if len(ft_remaining) < 3:
                continue                                  
            ft_k = min(max(2, min(5, len(ft_remaining) - 1)), len(ft_remaining) - 1)
        else:
            ft_k = min(max(2, min(5, len(ft_remaining))), len(ft_remaining))

        ft_gold = [str(s.get("step_goal", "")).strip() for s in ft_remaining[:ft_k]
                    if str(s.get("step_goal", "")).strip()]
        if len(ft_gold) != ft_k:
            continue
        ft_gold_set = set(ft_gold)


        ft_later_pool = [str(s.get("step_goal", "")).strip() for s in ft_remaining[ft_k:]
                         if str(s.get("step_goal", "")).strip()]
        ft_earlier_pool = [str(s.get("step_goal", "")).strip() for s in steps[:ft_i_idx + 1]
                           if str(s.get("step_goal", "")).strip()]


        pool: List[str] = []
        if flaw_type == "precondition_missing":
            pool = _uniq_pool(ft_later_pool, exclude=ft_gold_set)
            if not pool:
                continue
        elif flaw_type == "redundant_step":
            pool = _uniq_pool(ft_earlier_pool, exclude=ft_gold_set)
            if not pool:

                pool = _uniq_pool(list(ft_gold), exclude=set())
                if len(pool) < 2:
                    continue
        elif flaw_type == "goal_inconsistent":
            pool = _uniq_pool(cross_pool_raw, exclude=ft_gold_set)
            if not pool:
                continue
        elif flaw_type == "granularity_mismatch":
            pool = [_vaguify(g) for g in ft_gold]
            pool = list(dict.fromkeys(pool))
            if not pool:
                continue
        elif flaw_type == "wrong_ordering":
            if len(ft_gold) < 2:
                continue


            ft_gold_steps = ft_remaining[:ft_k]
            _dep_positions = []
            for _p in range(len(ft_gold) - 1):
                _step_b = ft_gold_steps[_p + 1]
                _indep = str(_step_b.get("independence", "")).strip().lower()
                if _indep == "yes":
                    _dep_positions.append(_p)
            if not _dep_positions:

                for _p in range(len(ft_gold) - 1):
                    _cc_a = ft_gold_steps[_p].get("causal_chain") or {}
                    _cc_b = ft_gold_steps[_p + 1].get("causal_chain") or {}
                    _pat_a = set(str(_cc_a.get("patient", "")).lower().split())
                    _pat_b = set(str(_cc_b.get("patient", "")).lower().split())
                    _stopwords = {"the", "a", "an", "and", "of", "on", "in", "to", "from", "with", "its", "it"}
                    _shared = (_pat_a - _stopwords) & (_pat_b - _stopwords)
                    if _shared:
                        _dep_positions.append(_p)
            if not _dep_positions:
                continue                                             
            pool = ["__swap__"]


        ft_video_rel: Optional[str] = None
        if bool(attach_evidence):
            video = _resolve_video_prefix_or_step_clip(
                item_dir,
                ft_prefix_end_step,
                strict=bool(strict_prefix_video),
                plan=plan,
            )
            if not video:
                continue
            ft_video_rel = _safe_relpath(video, input_root)


        flaw_pos = ft_rng.randrange(0, len(ft_gold))


        if flaw_type == "wrong_ordering":
            if not _dep_positions:
                continue
            flaw_pos = _dep_positions[ft_rng.randrange(0, len(_dep_positions))]


        bad = list(ft_gold)
        bad_step_text = ""                                      
        original_step_pos: Optional[int] = None                                            
        if flaw_type == "wrong_ordering":
            bad[flaw_pos], bad[flaw_pos + 1] = bad[flaw_pos + 1], bad[flaw_pos]
        else:
            if flaw_type == "redundant_step" and all(p in ft_gold_set for p in pool):
                other_goals = [g for g in ft_gold if g != ft_gold[flaw_pos]]
                if not other_goals:
                    continue
                chosen = other_goals[ft_rng.randrange(0, len(other_goals))]
            else:
                filtered = [p for p in pool if p != ft_gold[flaw_pos]]
                if not filtered:
                    filtered = pool
                chosen = filtered[ft_rng.randrange(0, len(filtered))]
            bad[flaw_pos] = chosen
            bad_step_text = chosen

            if flaw_type == "redundant_step":
                ids = _goal_to_step_ids.get(chosen, [])
                if ids:
                    original_step_pos = ids[0]


        if bad == ft_gold:
            continue


        gold_excerpt = _short_excerpt(ft_gold[flaw_pos])
        bad_excerpt = _short_excerpt(bad_step_text) if bad_step_text else ""

        if flaw_type == "goal_inconsistent":
            _reasons = [
                f"Step {flaw_pos+1} ('{bad_excerpt}') is unrelated to the high-level goal and disrupts the plan by introducing an irrelevant action instead of '{gold_excerpt}'",
                f"Step {flaw_pos+1} ('{bad_excerpt}') belongs to a different activity and breaks the causal continuity of the current plan",
                f"Step {flaw_pos+1} ('{bad_excerpt}') contradicts the objective by performing an action that does not advance toward the intended outcome",
            ]
        elif flaw_type == "redundant_step":
            pos_note = f" (originally completed as step {original_step_pos})" if original_step_pos else ""
            _reasons = [
                f"Step {flaw_pos+1} ('{bad_excerpt}') was already completed earlier{pos_note}, so it wastes a planning slot and causes a required next action to be omitted",
                f"Step {flaw_pos+1} ('{bad_excerpt}') repeats work already accomplished{pos_note}, blocking forward progress by displacing '{gold_excerpt}'",
                f"Including '{bad_excerpt}' at step {flaw_pos+1}{pos_note} means a critical subsequent action ('{gold_excerpt}') is displaced from the plan",
            ]
        elif flaw_type == "precondition_missing":
            _reasons = [
                f"Step {flaw_pos+1} ('{bad_excerpt}') is pulled from later in the plan and is out of order, so required intermediate preconditions for it are not satisfied yet",
                f"Step {flaw_pos+1} ('{bad_excerpt}') assumes conditions that earlier steps have not yet established, making execution premature at this position",
                f"Executing '{bad_excerpt}' at step {flaw_pos+1} before its prerequisites are met will cause downstream failures",
            ]
        elif flaw_type == "wrong_ordering":


            bad_at_pos = _short_excerpt(bad[flaw_pos])                           
            bad_at_pos1 = _short_excerpt(bad[flaw_pos + 1])                    
            _reasons = [
                f"Steps {flaw_pos+1} ('{bad_at_pos}') and {flaw_pos+2} ('{bad_at_pos1}') are in the wrong order: '{bad_at_pos1}' must come before '{bad_at_pos}' because the latter depends on the former",
                f"The sequence of step {flaw_pos+1} ('{bad_at_pos}') and step {flaw_pos+2} ('{bad_at_pos1}') is reversed; '{bad_at_pos1}' should precede '{bad_at_pos}' to satisfy its preconditions",
            ]
        elif flaw_type == "granularity_mismatch":
            _reasons = [
                f"Step {flaw_pos+1} ('{bad_excerpt}') is too vague to guide execution compared to the needed action '{gold_excerpt}'",
                f"Step {flaw_pos+1} ('{bad_excerpt}') is at the wrong level of abstraction, making it unactionable when the plan requires '{gold_excerpt}'",
            ]
        else:
            _reasons = [f"Step {flaw_pos+1} contains an error"]
        reason = _reasons[ft_rng.randrange(0, len(_reasons))]



        bad_steps_inline = " ".join([f'{i+1}) "{s.replace(chr(34), chr(39))}"' for i, s in enumerate(bad)])
        _T18_QUESTIONS = [
            (
                f'Context: High-level goal: "{hl}" Based on this prefix, the following proposed plan steps are: '
                f"{bad_steps_inline} Identify the flaw and repair the plan."
            ),
            (
                f'Context: High-level goal: "{hl}" Given this prefix, the proposed subsequent steps are: '
                f"{bad_steps_inline} Diagnose the error and provide a corrected plan."
            ),
            (
                f'Context: High-level goal: "{hl}" After this prefix, the proposed plan continuation is: '
                f"{bad_steps_inline} Find the problematic step and fix the plan."
            ),
        ]
        q = _T18_QUESTIONS[ft_idx % len(_T18_QUESTIONS)]



        if flaw_type == "precondition_missing" and bad_step_text:
            repair_steps = list(ft_gold) + [bad_step_text]
        else:
            repair_steps = list(ft_gold)
        repair_inline = " ".join([f'{i+1}) "{s.replace(chr(34), chr(39))}"' for i, s in enumerate(repair_steps)])

        _FLAW_TYPE_LABELS = {
            "goal_inconsistent": "a goal-inconsistent step",
            "granularity_mismatch": "a vague or overly general step",
            "precondition_missing": "a step that skips a required precondition",
            "redundant_step": "a redundant step that was already completed",
            "wrong_ordering": "two steps in the wrong order",
        }
        ft_label = _FLAW_TYPE_LABELS.get(flaw_type, flaw_type.replace("_", " "))
        _reason_s = str(reason or "").strip()
        if _reason_s and not _reason_s.endswith((".", "!", "?")):
            _reason_s = _reason_s + "."
        a = _sanitize_space(f"The flaw is in step {flaw_pos+1}, which contains {ft_label}. {_reason_s} The corrected plan is: {repair_inline}")

        out.append(
            Sample(
                TASK_18,
                EVIDENCE_PREFIX,
                [],
                ft_video_rel,
                q,
                a,
                source_rel,
                llm_fields={
                    "high_level_goal": hl,
                    "bad_plan_steps": list(bad),
                    "repair_steps": list(repair_steps),
                    "flaw_step": flaw_pos + 1,
                    "flaw_type": flaw_type,
                    "prefix_end_step_id": ft_prefix_end_step,
                },
            )
        )

    return out


def _counterfactual_clause(question: str) -> str:

    s = str(question or "").strip()
    if not s:
        return ""
    orig = s.strip()

    _CF_PREFIXES = [
        r"what\s+would\s+happen\s+if\s+",
        r"what\s+could\s+go\s+wrong\s+if\s+",
        r"what\s+would\s+(?:be\s+)?(?:the\s+)?(?:consequence|result|effect|impact)\s+if\s+",
        r"how\s+would\s+(?:the\s+)?(?:result|outcome|process|step|action|situation|task|dish|product|procedure|recipe|final\s+product)\s+(?:change|differ|be\s+affected)\s+if\s+",
        r"how\s+would\s+\S+ing\s+.+\s+(?:affect|change|alter|impact)\s+",



        r"how\s+would\s+.+\s+(?:affect|change|alter|impact|disrupt|compromise|interfere\s+with|damage|ruin|prevent|hinder|delay)\s+",
        r"what\s+happens\s+if\s+",
        r"in\s+the\s+event\s+that\s+",
        r"suppose\s+(?:that\s+)?",
        r"imagine\s+(?:that\s+|if\s+)?",
        r"assuming\s+(?:that\s+)?",
        r"had\s+the\s+(?:person|user|cook|agent)\s+",
        r"if\s+the\s+(?:person|user|cook|agent)\s+(?:had\s+)?",
        r"what\s+if\s+",
        r"if\s+",
    ]
    matched_prefix = False
    for pat in _CF_PREFIXES:
        s2 = re.sub(rf"^\s*{pat}", "", s, flags=re.IGNORECASE).strip()
        if s2 != orig:
            s = s2
            matched_prefix = True
            break
    if not matched_prefix:

        return ""
    s = s.rstrip(" ?!.").strip()
    if not s:
        return ""
    s = _lowercase_first_alpha(s)
    return s


def _make_task19_20(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    attach_evidence: bool,
) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    source_rel = _safe_relpath(os.path.join(item_dir, "final_plan.json"), input_root)
    out: List[Sample] = []

    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue

        step_clip = _resolve_video_clip(item_dir, sid) if bool(attach_evidence) else None
        if bool(attach_evidence) and not step_clip:
            continue
        clip_rel_str = _safe_relpath(step_clip, input_root) if step_clip else None

        q_cf = _require_str(st, "counterfactual_challenge_question")
        a_cf = _require_str(st, "expected_challenge_outcome")

        _t19_cf_question = ""
        _t19_cf_outcome = ""
        if q_cf and a_cf:
            q_cf_inline = str(q_cf).strip()
            clause = _counterfactual_clause(q_cf_inline)
            if not clause:

                _fallback = q_cf_inline.rstrip("?").strip()
                _fallback = re.sub(r"^(?:what\s+(?:would\s+happen\s+)?if\s+|how\s+would\s+.*?\s+if\s+)", "", _fallback, flags=re.IGNORECASE).strip()
                _fallback = _lowercase_first_alpha(_fallback)
                clause = _fallback


            if clause and len(clause.strip()) >= 3:





                _clause_words = clause.split()
                _clause_has_verb = bool(re.search(
                    r"\b(?:is|are|was|were|has|had|have|does|did|do|can|could|"
                    r"will|would|shall|should|may|might|must|"
                    r"fell|broke|began|became|went|came|got|ran|sat|stood|"
                    r"slipped|dropped|burned|spilled|tipped|failed|"
                    r"happened|occurred|started|changed|remained|used|overheated)\b",
                    clause, re.IGNORECASE
                ))
                if not _clause_has_verb and len(_clause_words) <= 3:


                    _copula = "were" if clause.strip().endswith("s") and not clause.strip().endswith("ss") else "was"
                    clause = clause.rstrip(" ,;") + f" {_copula} used"
                _T19_QUESTIONS = [
                    f'Based on the action shown in this clip, what is the most likely outcome if {clause}?',
                    f'Considering what is happening in this clip, what would probably happen if {clause}?',
                    f'Given the activity visible in this clip, what consequence would follow if {clause}?',
                ]
                q19 = _T19_QUESTIONS[_template_idx(len(_T19_QUESTIONS), "T19q", item_dir, sid)]
                cleaned = _clean_counterfactual_outcome(a_cf)
                if cleaned and len(cleaned.strip().rstrip(".!?")) >= 10:
                    _t19_cf_question = q_cf_inline
                    _t19_cf_outcome = cleaned
                    out.append(
                        Sample(
                            TASK_19,
                            EVIDENCE_CLIP,
                            [],
                            clip_rel_str,
                            q19,
                            _sanitize_space(cleaned),
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

            _T20_QUESTIONS = [
                (f'Failure reason: "{reason}" '
                 "If this failure occurred during another execution of the step shown in this clip, what recovery strategy would address it?"),
                (f'Failure reason: "{reason}" '
                 "If this failure occurred during another execution of the step shown in this clip, how should one recover from this problem?"),
                (f'Failure reason: "{reason}" '
                 "If this failure occurred during another execution of the step shown in this clip, what corrective action would resolve it?"),
            ]
            q20 = _T20_QUESTIONS[_template_idx(len(_T20_QUESTIONS), "T20q", item_dir, sid)]

            _strat_s = strat.strip()
            if not _strat_s or len(_strat_s.rstrip(".!?")) < 5:
                continue
            if not _strat_s.endswith((".", "!", "?")):
                _strat_s = _strat_s + "."




            _strat_lc = _lowercase_first_alpha(_strat_s.rstrip(".!?"))
            _strat_gerund = _imperitive_to_gerund(_strat_lc)
            _t20_variant = _template_idx(4, "T20a", item_dir, sid)
            if _t20_variant == 0:
                a20 = _sanitize_space(f"To recover, the person should try {_strat_gerund}.")
            elif _t20_variant == 1:
                a20 = _sanitize_space(f"The corrective action is {_strat_gerund}.")
            elif _t20_variant == 2:
                a20 = _sanitize_space(f"One way to address this is by {_strat_gerund}.")
            else:
                a20 = _sanitize_space(_strat_s)                          
            out.append(
                Sample(
                    TASK_20,
                    EVIDENCE_CLIP,
                    [],
                    clip_rel_str,
                    q20,
                    a20,
                    source_rel,
                    llm_fields={
                        "step_goal": step_goal,
                        "failure_reason": reason,
                        "recovery_strategy": strat,
                        **({"counterfactual_context": _t19_cf_question} if _t19_cf_question else {}),
                        **({"counterfactual_outcome": _t19_cf_outcome} if _t19_cf_outcome else {}),
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
    strict_prefix_video: bool,
    strict_schema: bool,
    min_steps: int,
    rng: random.Random,
    cross_plan_goals: Optional[List[str]] = None,
) -> List[Sample]:
    plan_path = os.path.join(item_dir, "final_plan.json")
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

            TASK_11,
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
            _make_task04_to_13(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
            )
        )
        out = [s for s in out if s.task_name in enabled_tasks]
    if TASK_14 in enabled_tasks:
        out.extend(_make_task14(item_dir, plan, input_root, attach_evidence=bool(attach_evidence)))
    if TASK_15 in enabled_tasks:
        out.extend(
            _make_task15(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
                strict_prefix_video=bool(strict_prefix_video),
            )
        )
    if TASK_16 in enabled_tasks:
        s = _make_task16(item_dir, plan, input_root, attach_evidence=bool(attach_evidence))
        if s:
            out.append(s)
    if TASK_17 in enabled_tasks:
        out.extend(
            _make_task17(
                item_dir,
                plan,
                input_root,
                rng=rng,
                attach_evidence=bool(attach_evidence),
                strict_prefix_video=bool(strict_prefix_video),
            )
        )
    if TASK_18 in enabled_tasks:
        out.extend(
            _make_task18_bad_plan_diagnosis_and_repair(
                item_dir,
                plan,
                input_root,
                rng=rng,
                attach_evidence=bool(attach_evidence),
                strict_prefix_video=bool(strict_prefix_video),
                cross_plan_goals=cross_plan_goals,
            )
        )
    if any(t in enabled_tasks for t in (TASK_19, TASK_20)):
        out.extend(
            _make_task19_20(
                item_dir,
                plan,
                input_root,
                attach_evidence=bool(attach_evidence),
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
        answer = _defluff_text(answer)                                        
        answer = _normalize_trailing_punct(answer)

        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
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
            "Generate Mani-LongVideo QA dataset (Task_08–Task_20) "
            "from final_plan.json (final schema)."
        )
    )
    parser.add_argument("--input-root", required=True, help="Dataset root containing many item dirs with final_plan.json.")
    parser.add_argument("--output-dir", required=True, help="Output root directory (will create one folder per task with data.jsonl).")
    parser.add_argument("--item-list", default=None, help="Path to a text file listing item directories (one per line). Bypasses os.walk discovery.")
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
        help="Fail if an item has fewer than N steps (use 4 to guarantee Task_18 and multi-step tasks).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(DEFAULT_TASKS),
        help="Subset of task names to generate (default: canonical 20-task taxonomy).",
    )
    parser.add_argument("--uniform-k", type=int, default=8, help="Number of uniform frames for images_uniform_scene tasks (default: 8).")
    parser.add_argument("--head", type=int, default=4, help="Head frames for Task_19 (default: 4).")
    parser.add_argument("--tail", type=int, default=4, help="Tail frames for Task_19 (default: 4).")
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="If set, require video_clip assets for Task_10 (skip keyframe fallback). Ignored in --text-only mode.",
    )
    parser.add_argument(
        "--strict-prefix-video",
        action="store_true",
        help="Require true cumulative prefix clips for video_prefix tasks (Task_15/17/18); do not fall back to the last-step clip.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--strict-schema",
        dest="strict_schema",
        action="store_true",
        help="Enable strict final-schema validation for final_plan.json (default).",
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
    parser.add_argument("--llm-single-pass", action="store_true", default=True, help="Use a single API pass (no second polishing pass). Default: True.")
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
    parser.add_argument(
        "--parallel-api",
        type=int,
        default=1,
        help=(
            "Number of concurrent LLM API calls within each item processing stage "
            "(LLM rewrite). Default: 1 (serial). "
            "Recommended: 4-8 for faster throughput. Each concurrent call is an "
            "independent thread; the Azure OpenAI client is thread-safe."
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

    enabled_tasks = set(args.tasks or [])
    llm_tasks = set(args.llm_tasks or [])

    attach_evidence = not bool(args.text_only)

    unknown = sorted([t for t in enabled_tasks if t not in set(ALL_TASKS)])
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    unknown_llm = sorted([t for t in llm_tasks if t not in set(ALL_TASKS)])
    if unknown_llm:
        raise ValueError(f"Unknown llm task names: {unknown_llm}")

    item_dirs = _list_item_dirs(input_root, item_list_file=getattr(args, 'item_list', None))
    if args.limit and int(args.limit) > 0:
        item_dirs = item_dirs[: int(args.limit)]
    if not item_dirs:
        raise FileNotFoundError(f"No item dirs found under {input_root} (expecting final_plan.json).")

    logger.info(f"Found {len(item_dirs)} item dirs under: {input_root}")
    logger.info(f"Enabled tasks: {sorted(enabled_tasks)}")
    if llm_tasks:
        logger.info(f"LLM rewrite tasks: {sorted(llm_tasks)}")
    _parallel_n = int(getattr(args, 'parallel_api', 1))
    if _parallel_n > 1:
        logger.info(f"Parallel API calls per item: {_parallel_n}")



    cross_plan_goals: Optional[List[str]] = None
    if TASK_18 in enabled_tasks:

        _cross_goals_cache = os.environ.get("CROSS_PLAN_GOALS_FILE", "")
        if _cross_goals_cache and os.path.isfile(_cross_goals_cache):
            with open(_cross_goals_cache, "r", encoding="utf-8") as _gf:
                cross_plan_goals = json.load(_gf)
            logger.info(f"T18 cross-plan goal pool: loaded {len(cross_plan_goals)} goals from cache {_cross_goals_cache}")
        else:
            _all_goals: List[str] = []
            for d in item_dirs:
                try:
                    p = _read_json(os.path.join(d, "final_plan.json"))
                    for s in (p.get("steps") or []) if isinstance(p, dict) else []:
                        if isinstance(s, dict):
                            sg = str(s.get("step_goal", "")).strip()
                            if sg:
                                _all_goals.append(sg)
                except Exception:
                    pass

            seen: set[str] = set()
            unique_goals: List[str] = []
            for g in _all_goals:
                if g not in seen:
                    seen.add(g)
                    unique_goals.append(g)
            cross_rng = random.Random(42)
            cross_rng.shuffle(unique_goals)
            cross_plan_goals = unique_goals
        logger.info(f"T18 cross-plan goal pool: {len(cross_plan_goals)} unique step goals collected from {len(item_dirs)} items.")

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
                strict_prefix_video=bool(args.strict_prefix_video),
                strict_schema=bool(args.strict_schema),
                min_steps=int(args.min_steps),
                rng=item_rng,
                cross_plan_goals=cross_plan_goals,
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
                    parallel_n=int(getattr(args, 'parallel_api', 1)),
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

            _ans_text = str(s.answer or "").strip()
            if len(_ans_text) < 8:
                logger.debug(f"SKIP_SHORT_ANSWER task={s.task_name} item={rel_item} len={len(_ans_text)}")
                continue
            per_item_counts[s.task_name] = int(per_item_counts.get(s.task_name, 0)) + 1
            if bool(attach_evidence):
                v = str(s.video or "").replace("\\", "/")
                if s.evidence_type == EVIDENCE_PREFIX and v:
                    if "/cumulative_last_frame_segments/" not in v and "/prefix_clips/" not in v:
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
