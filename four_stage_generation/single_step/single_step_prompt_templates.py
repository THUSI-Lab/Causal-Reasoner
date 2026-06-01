
from __future__ import annotations

from four_stage_prompt_templates import SYSTEM_PROMPT_ANALYST                           

from typing import Tuple


def build_stage1_user_prompt_single_step(num_frames: int, image_dimensions: Tuple[int, int]) -> str:

    w, h = image_dimensions
    return f"""
Analyze the provided {num_frames} frames (uniformly sampled from one continuous video, chronological order). Treat the frames as the ONLY source of truth.

Goal: Generate a step-by-step causal plan and step-level annotations for the entire video.

FULL-VIDEO COVERAGE (NON-NEGOTIABLE):
- The ordered `steps` MUST collectively cover the ENTIRE timeline from the FIRST frame to the LAST frame.
- The plan MUST NOT end early: the LAST step MUST include and reflect the last portion of the video (the last frames). Do NOT invent an "achieved final state" if the video ends mid-action; describe the last observed state and any visible ongoing action.
- Do NOT compress the whole plan into only the early/middle frames; later steps MUST reflect later-video events.
- Each step MUST correspond to a contiguous, localizable time interval in the video. Do NOT interleave events from different times inside the same step.

Language & grounding:
- Use objective, professional English.
- Do not hallucinate hidden states or off-screen objects.
- This is a SHORT video showing ONE coherent activity. You MUST produce exactly 1 step that covers the entire video from start to finish. Do NOT split it into multiple steps — the entire video is ONE step. Plans with more than 1 step are REJECTED automatically.
- Use consistent object naming across all fields (do not rename the same object with different synonyms).
- All required fields MUST be present and non-empty (no empty strings, empty arrays, empty objects, or null). In any string field (including list elements), do NOT reference frame/image indices, timestamps, durations, or timecodes. Avoid placeholders like "unknown", "N/A", "...".

FORMAT STANDARD (applies to all `causal_*` list fields in this output):
- Each `causal_*` field MUST be a JSON array of strings.
- Each string element MUST be a single, complete, objective English sentence grounded in the current step.
- Each string element MUST end with '.'.
- Each string element MUST NOT start with a list marker or numbering prefix (e.g., "1.", "2)", "-", "*", "•").
- Do NOT use newline characters inside any string element.
- MULTI-SENTENCE REQUIREMENT: each `causal_effect_on_*` list MUST contain at least 3 distinct sentences covering DIFFERENT aspects of the effect (e.g., one for the patient's state change, one for the spatial rearrangement, one for the functional consequence). Do NOT write one long run-on sentence using "When..., resulting in..., thereby..." — break into separate focused sentences.
- LIST SIZE: ALL `causal_*` lists (precondition and effect, spatial and affordance) MUST contain at least 3 sentences each, but aim for 4–5. Three is the MINIMUM, not the target — do not stop at 3 simply because the minimum is met. A well-annotated step typically needs 4–5 sentences per causal list to adequately cover: (1) the patient object's state, (2) the tool/agent hands and their contact configuration, (3) the workspace/supporting surfaces, (4) secondary objects affected by the action, and (5) environmental conditions or constraints. Omit a category only if genuinely inapplicable to this step.
- ANTI-TEMPLATE: Do NOT begin every sentence with the same syntactic pattern. Vary sentence openings.
- NO CROSS-FIELD REPETITION: Do NOT copy phrases verbatim between fields. Each field (precondition, effect, rationale, step_goal) must contribute UNIQUE information using DISTINCT vocabulary. If the same physical fact appears in both a precondition and an effect, describe it from different perspectives (e.g., precondition: "Package seal is intact." → effect: "Torn seal exposes a 5cm opening along the top edge.").
- SPATIAL vs AFFORDANCE SEPARATION (CRITICAL — apply to ALL causal_* fields):
  `causal_*_on_spatial` fields describe POSITIONAL RELATIONSHIPS BETWEEN objects: where objects are relative to each other (contact, support, containment, above/below/beside, distance, orientation of one object relative to another). Ask: "WHERE is object A relative to object B?"
  `causal_*_on_affordance` fields describe INTRINSIC OBJECT STATES AND PROPERTIES: the object's own functional/mechanical state (open/closed, sealed/unsealed, empty/full, graspable due to surface texture, wet/dry, hot/cold, separated/clumped, locked/unlocked). Ask: "WHAT state is this object in? What can it do?"
  If a statement mixes both (e.g., "jar is on counter and lid is open"), SPLIT it: spatial → "jar is on counter", affordance → "jar lid is in open position."
  NEVER put intrinsic state changes (sealed→unsealed, open→closed, assembled→disassembled) into spatial fields.
  NEVER put positional/support/containment relationships into affordance fields.
- IDLE/OBSERVATION BAN: Do NOT describe idle, pausing, watching, or waiting as part of the step goal or causal chain. If the video shows idle frames at the start or end, the step goal and causal chain should focus only on the goal-directed physical operations.
- Do NOT generate ANY keyframe-level fields.
- Do NOT reference frame/image indices or timestamps in any field.

TEMPORAL STRICTNESS (HARD CONSTRAINT — applies to ALL step-level causal_precondition_* and causal_effect_* fields):
- `causal_precondition_on_spatial` and `causal_precondition_on_affordance` describe the world state in the INSTANT BEFORE the step's action begins — as if you pressed pause on the video one frame before any movement starts. They are the NECESSARY enabling conditions, NOT descriptions of what happens during the action.
- `causal_effect_on_spatial` and `causal_effect_on_affordance` describe the world state in the INSTANT AFTER the step's action has FULLY COMPLETED — as if you pressed pause one frame after all movement has stopped and the new stable state is reached. They are the RESULTING states, NOT descriptions of what happens during the action.
- ANTI-MID-ACTION RULE: Do NOT describe states that exist only while the action is in progress (e.g., "hand is gripping the handle" as a precondition when the grip is established AS PART OF this step's action, or "object is being lifted" as an effect when the lifting IS the action). Preconditions are what must be true BEFORE the person starts moving; effects are what is true AFTER the person has finished moving and released.
- BAD (mid-action as precondition): "Hand is gripping the plate rim." — if gripping IS part of this step, this is mid-action, NOT a precondition.
- GOOD (true pre-action): "Plate is resting on the drying rack surface within arm's reach of the person's right hand." — this is the state BEFORE any reaching/gripping begins.
- BAD (mid-action as effect): "Hand is carrying the plate toward the cabinet." — carrying IS the action, NOT the result.
- GOOD (true post-action): "Plate is now resting on the cabinet shelf (was on the drying rack), and the person's hand has released contact with the plate rim." — this is the stable state AFTER all movement has stopped.

ACTION-RELEVANCE FILTER (applies to ALL fields):
- Every precondition, effect, and description sentence MUST be directly causally related to the action being performed. Omit background objects, ambient scene details, and elements not involved in or affected by the operation. Focus exclusively on the objects being manipulated, tools in use, and surfaces providing direct support/contact.

SPATIAL LINE REQUIREMENT:
 Each numbered line must explicitly name two entities and describe their visual spatial relationship.
 The relation must be directly observable from visual perception (e.g., geometric position, contact state, topological connection, relative placement).
 Avoid abstract or non-visual terms like "accessible/within reach/convenient" unless they can be grounded in measurable visual features (e.g., distance, reachability zone, field of view).
 Examples of valid visual relations: "object_a is on top of object_b", "object_a is inside container_c", "object_a is 10cm away from object_b", "object_a is aligned with the edge of object_b".

AFFORDANCE GROUNDING:
 Only include affordances that are directly visible or strongly implied by visible mechanical state (open/closed, sealed/unsealed, empty/full, free space available, grasped/not grasped, stable/unstable, separated/clumped).
 Do NOT assert hidden qualities (sharpness, cleanliness, "functional tap", "active heat") unless clearly visible.
 Affordances must describe operability states or functional states that enable/constrain the next physical action, NOT high-level semantic goals.
 VALID affordances (directly tied to physical manipulation):
  - Mechanical state: open/closed, sealed/unsealed, locked/unlocked, assembled/disassembled
  - Spatial occupancy: empty/full, has free space, blocked/unblocked
  - Physical state enabling manipulation: graspable/not graspable, stable/unstable, wet/dry surface (affects grip/sliding), hot/cold (affects touchability), separated/clumped (affects pickability)
  - Interaction-enabling configuration: stirrable configuration (object is submerged in container with liquid/semi-solid), pourable configuration (container has content and opening is unobstructed), cuttable configuration (object is stable on surface and blade can contact target)
 INVALID affordances (high-level goals or unverifiable qualities):
  - Semantic outcomes: "ready to be cooked", "prepared for serving", "maintains freshness", "evenly distributed"
  - Vague readiness/availability: "ready for X", "available for Y", "accessible for Z", "prepared for subsequent use", "within reach", "convenient for later"
  - Hidden material properties: "sharp enough", "clean", "non-stick surface", "functional heating element"
  - Taste/smell/texture qualities: "well-seasoned", "aromatic", "tender"
  - Environment side-effect summaries used as primary affordance: "countertop provides stable surface", "workspace remains available", "drawer remains accessible"
 EXCEPTION: Material properties are valid ONLY if visually confirmed in the frame (e.g., visible knife edge, visible rust/dirt, visible coating damage, visible steam indicating heat).

OBSERVABILITY RULE (for all precondition and effect fields — INTERNAL REASONING GUIDE, do NOT output DO/NDO labels in JSON):
 Every physical property or state you assert MUST be classified INTERNALLY as either DIRECTLY OBSERVABLE (DO) or NOT DIRECTLY OBSERVABLE (NDO):
 - DO (Directly Observable): position, contact, orientation, open/closed state, color, shape, gross motion, container level (full/empty), spatial arrangement — anything visible in the frame.
 - NDO (Not Directly Observable): internal temperature, internal pressure, chemical composition, sealed gas state, structural fatigue, exact weight, moisture content deep inside, flavor/taste.
 STRICT NDO RULE: Do NOT classify surface-visible properties as NDO. If you can SEE it in the frame (e.g., wet surface, visible steam, open lid, knife edge), it is DO, not NDO. NDO is reserved for truly invisible internal properties that require instruments to measure.
 Use this classification to FILTER your output: only assert DO properties freely; assert NDO properties ONLY if visually confirmed (e.g., visible steam confirms heat). Do NOT write "(DO)" or "(NDO)" tags in the output JSON.
 When writing `causal_effect_on_affordance`, follow this priority order:
 1. PRIMARY: State change of the patient (the acted-upon object) — e.g., "package seal is broken (contents now extractable)", "onion outer layer is separated from flesh"
 2. SECONDARY: State change of the tool/agent contact object — e.g., "knife blade retains cutting capability"
 3. TERTIARY (only if space permits): Environment/workspace side effects — e.g., "cutting board has less free space"
 Do NOT write only tertiary effects. Every `causal_effect_on_affordance` list MUST begin with at least one primary effect.

Examples (contrast; follow the GOOD style):
SPATIAL examples:
- Bad: "Ingredients are accessible on the counter."
  Good: [
    "Chopped onion is on cutting board.",
    "Cutting board is on counter surface.",
    "Knife is 15cm to the right of cutting board."
  ]
AFFORDANCE examples:
- Bad: "The ingredients are ready to be cooked."
  Good: [
    "Chopped vegetables are inside pan (stirrable configuration: submerged in oil).",
    "Pan has free space of 3cm from rim (allows stirring without spillage)."
  ]

- Bad: "The knife is sharp and clean."
  Good: [
    "Knife blade edge is visible and intact (capable of cutting).",
    "Knife is stable on cutting board (graspable without slipping)."
  ]

- Bad: "The countertop will continue to provide a stable surface for subsequent steps."
  Good: [
    "Package seal is torn open along top edge (onions inside are now extractable by hand).",
    "Package plastic retains structural integrity (can still contain remaining onions)."
  ]

ENTITY CONSISTENCY (NON-NEGOTIABLE):
 `patient` must be exactly one entity id. Format: use spaces to separate words within an entity name, use colons to separate different entities.
 Example: "chopped onion" (single entity), "cutting board:sharp knife" (two entities).
 Do not concatenate multiple objects into one id. Mention secondary entities inside the causal_* strings.
 GLOBAL ENTITY REGISTRY: Establish a FIXED base name for each object (e.g., "onion", "serrated knife", "large cutting board"). Use that EXACT name in ALL fields — `patient`, `agent`, `causal_*` sentences, `step_goal`, `rationale`, `counterfactual_challenge_question`, `failure_reflecting`. When an object changes state (cut, opened, cooked), keep the SAME base name — describe the transformation in causal_effect fields, NOT in the patient name.
 Bad: patient="whole_tomato" early, then patient="opened_tomato_sections" later
 Good: patient="tomato" everywhere. Describe state changes in causal_effect.
 Bad: patient="sealed_package", then "opened_package"
 Good: patient="onion_package" everywhere.

Output format (strict JSON only; no extra text):
{{
  "high_level_goal": "One comprehensive English sentence describing the overall goal and intended final outcome of the entire video. This should capture ALL major activity phases visible from start to finish.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "One or two concise English sentences that COMPREHENSIVELY summarize ALL major actions performed during this video — from start to finish. COVERAGE RULE: every distinct physical sub-task visible in the video MUST be mentioned. VERB FORM: use base/infinitive verb form consistently (e.g., 'Peel the outer skin off the onion' NOT 'Peeling...' or 'Peeled...' or 'Peels...').",
      "rationale": "Grounded sentences explaining WHY this step (the sole step) is performed — what physical/spatial/affordance preconditions it assumes and what effects it produces. CAUSAL LANGUAGE: MUST use causal connectives (because, since, therefore, in order to, which enables, which leads to). Do NOT just restate step_goal. Explain the PHYSICAL dependency chain.",
      "causal_chain": {{
        "agent": "Primary force/controller (prefer body part like 'hands'/'right hand'; use a tool part only if it is clearly the direct force applicator). NATURAL LANGUAGE: use spaces not underscores (e.g., 'right hand' not 'right_hand').",
        "action": "Verb phrase summarizing the core physical action (include the physical mechanism when helpful; e.g., 'apply torque to loosen', 'tilt to pour'). BANNED VAGUE VERBS: do, use, handle, manipulate, interact with, work on, manage, deal with, process, operate, arrange, organize, prepare, set up, observe, watch, wait, pause, stand, idle, rest, monitor. Use specific physical verbs instead: push, pull, rotate, tilt, insert, press, grasp, lift, lower, place, release, slide, pour, cut, peel, tear, fold, unfold, screw, unscrew, wipe, rinse, squeeze, stir, shake, flip, tap, align, withdraw, stabilize, adjust, carry, transport.",
        "patient": "Primary entity being acted upon (use spaces between words, e.g. 'dirty plate', 'rice cooker pot').",
        "causal_precondition_on_spatial": "MACRO spatial precondition statements — describe spatial relations that MUST ALREADY HOLD BEFORE this action begins. (FORMAT STANDARD). (TEMPORAL STRICTNESS: describe ONLY the state that exists BEFORE ANY action begins — not mid-action states.)",
        "causal_precondition_on_affordance": "MACRO affordance/state precondition statements — describe functional/material/state properties of objects that MUST ALREADY HOLD BEFORE this action begins. SPECIFICITY REQUIREMENT: BANNED standalone terms — do NOT write just 'graspable', 'pourable', 'cuttable' alone. ALWAYS state the PHYSICAL PROPERTY that enables the affordance. (FORMAT STANDARD). (TEMPORAL STRICTNESS.)",
        "causal_effect_on_spatial": "MACRO spatial effect statements AFTER the action completes — describe the KEY spatial changes to the primary patient object. STATE-CHANGE LANGUAGE: use explicit transition markers showing BEFORE->AFTER change. (FORMAT STANDARD). (TEMPORAL STRICTNESS.)",
        "causal_effect_on_affordance": "MACRO affordance/state effect statements AFTER the action completes — MUST follow the AFFORDANCE EFFECT HIERARCHY: start with the patient's core functional state change, then tool state if relevant, then environment only if essential. (FORMAT STANDARD). (TEMPORAL STRICTNESS.)"
      }},
      "counterfactual_challenge_question": "One realistic counterfactual what-if question that could disrupt this action due to physics/constraints, grounded in the scene. MUST start with 'What if ...?'. The what-if MUST target a SPECIFIC physical/spatial/affordance condition involving a VISIBLE object or relation in the current scene.",
      "expected_challenge_outcome": "Predicted physical outcome if that counterfactual challenge occurs. MUST be ONE single, specific, immediate physical consequence grounded in this step's spatial setup and affordances. SECOND-ORDER REASONING: after stating the immediate consequence, explain what downstream task outcome this would prevent or alter.",
      "failure_reflecting": {{
        "reason": "Most plausible real (non-counterfactual) failure mode. SEVERITY REQUIREMENT: the failure MUST substantially block or derail completion. GROUNDING REQUIREMENT: based on visible physical conditions. SPECIFICITY REQUIREMENT: MUST name the SPECIFIC object(s) and SPECIFIC physical mechanism.",
        "recovery_strategy": "ONE concrete, physically plausible recovery action. SAFETY REQUIREMENT: recovery must be safe. MINIMAL RECOVERY: only restore the specific condition broken by the failure."
      }}
    }}
  ]
}}

Additional constraints:
- `step_id` MUST be 1 (exactly one step).
- The `step_goal` must be specific and comprehensive, covering ALL actions visible in the video.
- CAUSAL CHAIN COMPLETENESS (IMPORTANT):
  The causal annotation MUST cover ALL FOUR components of a complete causal chain:
  1. SPATIAL SETUP: the spatial arrangement enabling the interaction (in causal_precondition_on_spatial)
  2. AFFORDANCE MECHANISM: the functional property enabling the force/action (in causal_precondition_on_affordance)
  3. FORCE/ACTION APPLICATION: how force is transferred (captured in action + agent + rationale)
  4. CONCRETE RESULT: the specific, named state change of the patient object (in causal_effect_on_*)
- Do NOT add any extra keys beyond the schema above.
- The step should be anchorable to visual evidence.


Now output the final strict JSON object only.
""".strip()
