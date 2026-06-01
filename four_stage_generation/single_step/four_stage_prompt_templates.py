from __future__ import annotations

from typing import Tuple


SYSTEM_PROMPT_ANALYST = """
You are a highly advanced AI acting as a Physical Interaction Analyst and Causal Planner. Your primary mission is to deconstruct observed actions in video frames into their fundamental causal, spatial, and affordance-based physical principles.
You must analyze key moments from a continuous action sequence to produce structured annotations grounded strictly in visual evidence.
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
Ensure outputs cover the entire video timeline from the first provided frame to the last provided frame.
DARK/CORRUPTED FRAME GUARD: If the majority of provided frames are entirely dark, black, heavily occluded (e.g., lens cap on, viewfinder artifacts), or show no discernible kitchen/workspace scene, output a JSON object with `"error": "no_visual_content"` and `"reason": "Majority of frames are dark/black/corrupted with no visible activity."` instead of generating a plan. Do NOT hallucinate actions or objects from featureless frames.
Core definitions (use consistently in ALL causal fields):
1) Spatial Relations (Geometric/Topological): Define the visible positional relationships BETWEEN objects.
   - Preconditions/Effects: Define contact (touching, resting-on), relative position (inside, on_top_of, beside, above, below), containment, support relations, orientation of one object relative to another.
   - Focus: WHERE are objects physically placed relative to each other? What geometric/topological relationships hold?
2) Affordances (Functional/Intrinsic States): Define the object's OWN intrinsic state, properties, and physical mechanisms.
   - Preconditions/Effects: Define the object's mechanical state (open/closed, sealed/unsealed, locked/unlocked, assembled/disassembled), material properties (elastic, spreadable, dry/wet surface), functional readiness based on intrinsic properties (graspable due to texture, pourable because opening is unobstructed, cuttable because edge is intact).
   - Focus: WHAT state is this object in? What are its intrinsic physical properties? What can it do based on its own state? (NOT where it is relative to other objects — that is spatial.)
   - HIERARCHY: Always prioritize the PRIMARY acted-on object's functional state change over secondary objects, tools, or workspace surfaces. Environment-level side effects (countertop availability, storage space) are lowest priority and should only appear after all primary object states are covered.
   - BAN on vague readiness language: Do NOT write "ready for X", "available for Y", "accessible for Z", or "prepared for subsequent use" as standalone affordance statements. Instead, state the SPECIFIC mechanical/functional state change (e.g., "seal is broken, exposing contents" instead of "package is ready for use").
3) Action-Relevance Filter (applies to ALL annotation fields):
   - Every statement in preconditions, effects, rationale, and descriptions MUST be strictly relevant to the physical operation being performed.
   - INCLUDE: objects directly manipulated, tools used, surfaces providing direct support/contact for the action, containers/receptacles involved, body parts executing the action.
   - EXCLUDE: background furniture not involved in the action, ambient lighting/weather, other people not participating, decorative items, general room layout, objects that happen to be visible but are not causally connected to the operation.
   - SELF-CHECK: For every sentence you write, ask "Would removing this object/state change whether the action succeeds or fails?" If NO, omit the sentence.

MATERIAL HALLUCINATION RULE (applies to ALL text fields in ALL stages — patient, step_goal, rationale, caption, action, causal_* sentences):
 Do NOT assert specific material names (brass, chrome, stainless steel, oak, marble, copper, aluminum, ceramic, porcelain, iron, granite, bamboo, teak, walnut, mahogany, bronze, pewter, etc.) unless the material is UNAMBIGUOUSLY identifiable from visual appearance alone.
 Use generic, visually-grounded descriptions instead:
  - "metal handle" not "brass handle"; "metal-colored handle" not "brass-colored handle"
  - "dark wooden board" not "oak cutting board"; "light wooden spoon" not "bamboo spoon"
  - "white bowl" not "porcelain bowl"; "white plate" not "ceramic plate"
 Color, shape, texture, and finish (matte/glossy/ridged) are observable; exact material composition is NOT.
 COLOR CAUTION: Do NOT include color in object names unless needed to distinguish two same-type objects in the scene. Prefer size/shape/function names: "large knife", "serrated knife", "small cutting board" over "red-handled knife", "blue cutting board". Video compression and lighting distort colors.
 EXCEPTION: Transparent materials (glass, clear plastic) and obviously identifiable materials (paper, cardboard, fabric/cloth) may be named when visually unambiguous.

CAPTION QUALITY TRIAD (applies to step_goal, action_state_change_description, and caption fields across ALL stages):
 Every descriptive text field MUST address all three components:
 1. SPATIAL: Where are the key objects relative to each other at the start and/or end of the action?
 2. MOTION: What physical motion, force, or manipulation is applied (direction, trajectory, mechanism)?
 3. STATE CHANGE: What observable property transitions from state_A to state_B (contact gained/lost, open/closed, grasped/released, supported/unsupported, inside/outside, assembled/separated)?
 For TRANSITION actions (reach, carry, walk) where no object state changes, describe the SPATIAL PROGRESSION and what CONTACT or PROXIMITY state changes (e.g., "hand transitions from resting on counter to hovering above the cabinet handle").
 SELF-CHECK: before finalizing any caption/description, verify all three components are present. If any is missing, rewrite.

SPATIAL vs AFFORDANCE DECISION RULE (use when populating causal_* fields):
 - If the statement describes WHERE objects are relative to each other (position, contact, containment, support) → SPATIAL
 - If the statement describes WHAT an object can do or what functional/mechanical state it is in (open/closed, graspable, pourable, sealed/unsealed) → AFFORDANCE
 - If both position AND function are mixed in one statement → SPLIT into two separate statements, one per category
 - COMMON MISTAKE — these belong in SPATIAL, not affordance: containment ("X is inside Y", "pan still holds all the vegetables"), arrangement/distribution ("strips are spread in a single layer", "sauce covers the left half"), layering ("noodles are intermixed with vegetables"), occlusion/visibility ("hidden under the solids", "exposed on the base"), positional stability ("pan remains stabilized against rotation"). These all describe WHERE things are relative to each other.
 Example: "The jar is on the counter and its lid is open" → spatial: "The jar is on the counter." + affordance: "The jar lid is in the open position, exposing the interior."
""".strip()

def build_stage1_user_prompt(num_frames: int, image_dimensions: Tuple[int, int]) -> str:
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
- SHORT VIDEO CONSTRAINT (HARD): These are SHORT videos (under 20 seconds). Use EXACTLY 1 step. The entire video activity MUST be described as a SINGLE step covering the full timeline from first frame to last frame. Plans with more than 1 step are rejected automatically. Do NOT segment a short clip into multiple steps under any circumstances.
- Step guidance:
  - ONE STEP ONLY: Describe the entire video as one coherent activity in a single step, even if multiple hand movements or sub-actions occur.
  - IDLE/OBSERVATION BAN: Do NOT create steps for idle, pausing, watching, or waiting periods. Every step MUST involve visible physical manipulation of an object. BANNED primary step actions: observe, watch, wait, pause, stand, idle, rest, monitor.
- Use consistent object naming across all steps (do not rename the same object with different synonyms).
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
  - Container/volume state: empty/full, has free space, blocked/unblocked
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
AFFORDANCE examples (intrinsic object states/properties — NOT positional relationships):
- Bad: "The ingredients are ready to be cooked."
  Good: [
    "Chopped vegetables are in stirrable configuration (pieces submerged in oil, movable by implement).",
    "Pan interior has remaining capacity (not full to rim, allowing stirring without spillage)."
  ]

- Bad: "The knife is sharp and clean."
  Good: [
    "Knife blade edge is visible and intact (capable of cutting).",
    "Knife handle has dry textured surface (provides friction for secure grip)."
  ]

- Bad: "The countertop will continue to provide a stable surface for subsequent steps."
  Good: [
    "Package seal is torn open along top edge (onions inside are now extractable by hand).",
    "Package plastic retains structural integrity (can still contain remaining onions)."
  ]

ENTITY CONSISTENCY (NON-NEGOTIABLE):
 `patient` must be exactly one entity id. Spaces between words, colons between entities.
 Example: "chopped onion" (single entity), "cutting board:sharp knife" (two entities).
 Do not concatenate multiple objects into one id. Mention secondary entities inside the causal_* strings.
 GLOBAL ENTITY REGISTRY: Establish a FIXED base name for each object (e.g., "onion", "serrated knife", "large cutting board"). Use that EXACT name in ALL fields across ALL steps — `patient`, `agent`, `causal_*` sentences, `step_goal`, `rationale`. When an object changes state (cut, opened, cooked), keep the SAME base name — describe the transformation in causal_effect fields, NOT in the patient name.
 Bad: Step 1 patient="whole_tomato", Step 3 patient="opened_tomato_sections"
 Good: patient="tomato" in ALL steps. Describe state changes in causal_effect.
 Bad: Step 1 patient="sealed_package", Step 2 "opened_package"
 Good: patient="onion_package" in ALL steps.

Output format (strict JSON only; no extra text):
{{
  "high_level_goal": "One comprehensive sentence covering ALL major activity phases in the video (preparatory, main, concluding). Use subordinating structures (after/by/while) to link phases naturally. Self-check: list all visible phases, verify each is represented.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "One or two concise sentences describing ALL major actions in this step. Must correspond to exactly one continuous physical sub-task. Use base/infinitive verb form. Do NOT include actions from adjacent steps.",
      "rationale": "One sentence explaining this step's specific physical role in achieving the high_level_goal — why the overall plan would be incomplete without it. Do NOT restate step_goal or use generic justifications.",
      "causal_chain": {{
        "agent": "Primary force/controller (prefer body part; use tool only if it is the direct force applicator).",
        "action": "Verb phrase summarizing the core physical action (include mechanism when helpful, e.g., 'apply torque to loosen'). BANNED VAGUE VERBS: do, use, handle, manipulate, interact with, work on, manage, deal with, process, operate, arrange, organize, prepare, set up, observe, watch, wait, pause, stand, idle, rest, monitor.",
        "patient": "Primary entity acted upon (use spaces between words; reuse same identifier across all steps).",
        "causal_precondition_on_spatial": "Positional relationships BETWEEN objects that must hold BEFORE the step begins. Name two entities per statement. No intrinsic states here.",
        "causal_precondition_on_affordance": "Intrinsic object states/properties that must hold BEFORE the step begins. State the physical property enabling each affordance (e.g., 'dry textured surface provides friction for grip' not just 'graspable'). No positional relationships here.",
        "causal_effect_on_spatial": "Positional changes AFTER the step completes. Use BEFORE→AFTER transition markers (e.g., 'Onion is now inside the pan (was on the cutting board)'). No intrinsic state changes here.",
        "causal_effect_on_affordance": "Intrinsic state changes AFTER the step completes. Start with patient's core state change, then tool, then environment only if essential. No 'ready for X' language."
      }},
      "counterfactual_challenge_question": "One realistic 'What if ...?' question targeting a SPECIFIC visible physical/spatial/affordance condition that could disrupt this step.",
      "expected_challenge_outcome": "The immediate physical consequence, plus what downstream task outcome it would prevent. Do NOT propose recovery actions or generic outcomes.",
      "failure_reflecting": {{
        "reason": "Most plausible real failure mode that would substantially block step completion. Name the SPECIFIC object and physical mechanism. Should target the same physical vulnerability domain as the counterfactual when possible.",
        "recovery_strategy": "ONE concrete physical maneuver to restore the broken condition. Name the specific object and describe the action in enough detail to execute. Use DIFFERENT vocabulary than the failure reason."
      }}
    }}
  ]
}}

Additional constraints:
- Step ordering MUST follow chronological frame order. `step_id` starts at 1, increments by 1.
- Each `step_goal` must be specific, non-duplicated, and cover ALL sub-tasks within its time span.
- STEP COHERENCE: Each step = one continuous physical sequence targeting the same primary object/workspace. Split independent sub-tasks (different objects, different areas) into separate steps.
- TEMPORAL BLEED PROHIBITION: step_goal MUST NOT describe actions from adjacent steps.
- GRANULARITY: Use exactly 1 step for these short videos. The single step covers the entire activity.
- CAUSAL CHAIN COMPLETENESS: Every step needs all four components — (1) spatial setup, (2) affordance mechanism, (3) force/action, (4) concrete result naming the patient's specific state change. The RESULT component is most commonly omitted — verify it.
- CROSS-STEP CONSISTENCY:
  (a) Step i's effects MUST make Step i+1's preconditions plausible.
  (b) At least ONE effect of Step i must be a KEY enabling precondition for Step i+1 (functional state change preferred over trivial positional continuity).
  (c) Physical state continuity: if Step i ends with object grasped/held, Step i+1 must not claim it's resting on a surface. Hand identity (left/right) must be consistent across boundaries.
  (d) Effect and precondition sentences MUST use different vocabulary (no verbatim copies). Use causal direction markers ("enabling", "allowing") in effect text.
  (e) "No dependency" answers are PROHIBITED — every step (except step 1) depends on at least one prior effect.
- Do NOT add extra keys beyond the schema.
- Each step should be anchorable to visual evidence.


Now output the final strict JSON object only.
""".strip()

def build_stage2_user_prompt(high_level_goal: str, draft_plan_outline: str, num_frames: int) -> str:
    return f"""
You are an expert video step temporal localization assistant.
You are given:
1) {num_frames} uniformly sampled frames from the FULL original video (chronological order).
2) A draft step list extracted from a plan (read-only; do NOT edit it).

High-level goal (context): {high_level_goal}

Draft steps (read-only):
{draft_plan_outline}

Note on indices:
- Some frames may look identical due to uniform sampling/padding; avoid choosing a segment whose boundaries fall on visually identical frames with no time progress.

Task:
For EACH step, predict the corresponding time interval in the original video by selecting:
- `start_frame_index`: the 1-based index of the boundary timestamp where this step starts (inclusive).
- `end_frame_index`: the 1-based index of the boundary timestamp where this step ends (exclusive; the first frame AFTER the step ends).

Interpretation:
- Let `t(i)` be the timestamp of sampled frame `i`.
- The step clip is cut as the half-open interval `[t(start_frame_index), t(end_frame_index))`.
- Because boundaries are on a shared grid, `end_i` may equal `start_(i+1)` (contiguous, no overlap).

Procedure (MANDATORY TWO-PASS BOUNDARY DETERMINATION):

PASS 1 — Identify step completion moments:
For EACH step, scan the frames to find the LAST frame where this step's action is STILL IN PROGRESS or has JUST COMPLETED. This is the "completion frame" — the frame where:
  - The hand/tool has fully released the object AND the object is in its final resting position for this step, OR
  - The body has returned to a neutral/transition pose before starting the next action, OR
  - The described world-state change has become fully visible and stable.

PASS 2 — Identify next-step initiation moments:
For EACH step (except the last), find the FIRST frame where the NEXT step's action has CLEARLY BEGUN. This is the "initiation frame" — the frame where:
  - The hand/tool has begun reaching toward the NEXT step's target object, OR
  - A new grip/contact is being established on a DIFFERENT object, OR
  - The agent's body is clearly oriented toward the next task's workspace.

BOUNDARY PLACEMENT RULE:
- The boundary `end_frame_index` for step i MUST be set so that frame `end_frame_index - 1` (the LAST frame included in step i's clip) shows step i's action COMPLETED with NO visible motion toward step i+1's target.
- Specifically: `end_frame_index` = completion_frame + 1, where completion_frame is the last frame showing step i completed.
- ANTI-LEAKAGE TEST: Before finalizing each boundary, mentally check: "In frame end_frame_index - 1, is the agent's hand/body already moving toward the next step's object?" If YES, decrease end_frame_index by 1 and re-check.
- If there are idle/transition frames between completion and initiation, assign them to step i (the completed step). HOWEVER: if the idle frames show the agent already re-orienting toward step i+1's workspace, they belong to step i+1 instead.
- NO-PAD RULE: Do NOT pad extra frames "just in case" — that causes next-step bleeding.

ASYMMETRIC ERROR POLICY (NON-NEGOTIABLE):
- It is MUCH WORSE to include next-step frames in the current step's clip than to lose a frame at the tail of the current step.
- When uncertain, prefer ending the current step EARLIER (even if you lose the last 1-2 frames of the step's action) rather than LATER (which risks including the beginning of the next step).
- Specifically: if you cannot determine whether frame F shows the current step completing or the next step beginning, assign frame F to the NEXT step (set end_frame_index of current step to F, not F+1).

WHAT CONSTITUTES A STEP BOUNDARY (visual cues to look for):
1. RELEASE-REACH transition: The hand releases the current object (fingers open, no contact), and then begins reaching toward a new object. The boundary is AFTER the release is complete but BEFORE any reaching toward the next object begins.
2. PLACEMENT-WITHDRAWAL transition: An object is placed in its final position (no longer moving), and the hand begins withdrawing. The boundary is AFTER the object is stationary.
3. TOOL CHANGE: A tool is put down and a different tool is picked up. The boundary is AFTER the first tool is released.
4. WORKSPACE SHIFT: The agent's focus/body orientation shifts from one area to another. The boundary is at the moment of shift.
5. POSE RESET: The agent returns to a neutral stance between actions. The boundary is at the neutral pose.

STEP DEPENDENCY (quick judgment alongside boundaries):
For each step except Step 1, set `independence` to `"yes"` if the previous step physically enables this one (e.g., an object moved/opened/created that this step needs), or `"no"` otherwise.
Do NOT include `independence` for Step 1.

IMPORTANT:
- All required keys MUST be present. For step_id >= 2, the `independence` field (string) is also required alongside the integer boundary fields.
- Indices refer ONLY to the provided {num_frames} frames (1..{num_frames}).
- You MAY output `{num_frames + 1}` ONLY for `end_frame_index` to indicate the exclusive boundary AFTER the last provided frame (typically for the last step to cover the video end).
- Output must contain exactly one entry per draft `step_id` and MUST NOT include any extra step_ids.
- Enforce monotonic, contiguous, non-overlapping segments (no gaps): for consecutive steps, `end_i == start_(i+1)`.
- Do NOT leave uncovered time between steps; if uncertain, choose boundaries that preserve full coverage rather than risking missing late-stage events.
- HARD full-video coverage (NON-NEGOTIABLE): `start_1` MUST be `1` and `end_last` MUST be `{num_frames + 1}` (use `{num_frames + 1}` to indicate the exclusive boundary AFTER the last provided frame). Do NOT end early.
- BALANCE CHECK: The single step covers the full video. It must span all frames.
- BOUNDARY ACCURACY (NON-NEGOTIABLE): Choose `start_frame_index` / `end_frame_index` so each step clip is accurate, complete, and rigorous:
  - Each step clip [start_frame_index, end_frame_index) MUST contain the COMPLETE execution of that step's action to its conclusion (the step's END STATE must be visible in the clip).
  - Each step clip MUST NOT contain frames showing the NEXT step's action in progress (no next-step reaching, gripping, or manipulating a different object).
  - Prefer the smallest interval that fully contains the step. Do NOT pad extra frames "just in case" — that causes next-step bleeding.
  - When the step ends with a release/placement, the boundary should capture the release completion but exclude any subsequent reaching toward a new object.
- Output MUST be exactly one JSON object with a single top-level key `steps` (no other top-level keys).

Output format (strict JSON only):

Field definitions (read carefully; output JSON must contain ONLY the keys in the template):
- `steps` (list): Exactly one entry per draft `step_id` (no extra/missing ids), in ascending `step_id` order.
- `steps[*].step_id` (int): Draft step identifier (must match exactly; do not renumber/reorder).
- `steps[*].start_frame_index` (int): Inclusive start boundary (1-based, within [1, {num_frames}]). Choose the boundary where the step begins; when uncertain, bias slightly earlier to preserve context for Stage 3.
- `steps[*].end_frame_index` (int): Exclusive end boundary (1-based, within [2, {num_frames + 1}]). Choose the first boundary AFTER the step's action has fully completed AND BEFORE any next-step action begins. Must satisfy `start_frame_index < end_frame_index`. Use `{num_frames + 1}` for the last step.
- `steps[*].independence` (string, ONLY for step_id >= 2): `"yes"` or `"no"`. Do NOT include for Step 1.

Output JSON template (replace the numbers with your chosen indices; keep keys exactly):
{{
  "steps": [
    {{
      "step_id": 1,
      "start_frame_index": 1,
      "end_frame_index": 2
    }},
    {{
      "step_id": 2,
      "start_frame_index": 2,
      "end_frame_index": 5,
      "independence": "yes"
    }}
  ]
}}
""".strip()


def build_stage3_user_prompt(high_level_goal: str, draft_plan_outline: str, draft_step_json: str, num_frames: int) -> str:
    return f"""
You are an expert Physical Interaction Analyst and Causal Planner.
You are given {num_frames} uniformly sampled frames from a SINGLE STEP CLIP (chronological order), and the draft step definition (read-only).

DARK/CORRUPTED FRAME GUARD: If the majority of these step-clip frames are entirely dark, black, heavily occluded, or show no discernible activity, output a JSON object with `"error": "no_visual_content"` and `"reason": "Majority of step-clip frames are dark/black/corrupted with no visible activity."` instead of generating annotations. Do NOT hallucinate actions or objects from featureless frames.

Task:
Using the step-clip frames as the PRIMARY evidence, refine and complete the annotation for this step and generate 2 keyframe annotations.

Keyframe selection procedure (recommended; follow silently):
1) Scan all frames quickly to understand the step progression and physical state changes.
2) Pick exactly 2 DISTINCT frames that are the two most causally important and visually anchorable key moments within this step (NOT limited to initiation/completion).
3) Treat each keyframe as a conjunction of constraints: the selected `frame_index` MUST be consistent with its own
   `action_state_change_description`, `causal_chain` (frame-level), and `interaction` simultaneously (avoid partial matches).
4) Do an explicit self-check BEFORE you finalize: for each selected `frame_index`, every factual claim in the corresponding
   `critical_frames[*]` object MUST be visually grounded in that exact image (preconditions, contacts, spatial relations, object identities).
   If a mismatch remains, FIX IT NOW by revising the text and/or selecting a different `frame_index` (do NOT defer mismatches to a later pass).
5) Ensure the 2 selected frames are in chronological order (`frame_index` strictly increases). If multiple frames match similarly well, break ties by **key-moment fidelity** (NOT by being early/late in the clip):
   - Prefer the frame where the described micro-action / state-change is most visually evident and discriminative.
   - Avoid idle/paused frames if there exists a frame that shows the action or decisive state change more clearly.
   - If the step's outcome persists across many frames, prefer the earliest frame where that outcome becomes true and stable (or the clearest transition), rather than a later static frame.
6) DISTRIBUTION CONSTRAINT (HARD RULE — ZERO TOLERANCE):
   - The 2 critical frames MUST NOT both fall in the last 25%% of the step clip (i.e., both frame_index > {num_frames} * 0.75 is REJECTED).
   - The gap between the two frame_index values MUST be at least 15%% of {num_frames} (i.e., frame_index_2 - frame_index_1 >= {max(2, int(num_frames * 0.15))}). If both frames are clustered within a narrow range, one of them is likely redundant — pick a more informative frame from a different phase of the step.
   - RATIONALE: Critical frames should capture TWO DISTINCT phases of the step's physical progression. If a step has an opening phase (approach/grasp) and a closing phase (place/release), each CF should represent one of these phases — not two nearby frames from the same moment.
   - SELF-CHECK: after selecting both frames, verify (a) they are not both in the last quarter, and (b) they are separated by at least {max(2, int(num_frames * 0.15))} frames. If either check fails, move the earlier frame to a more representative moment in the first half of the clip.

Strict requirements:
- You MUST NOT change `step_id` from the draft.
- You MAY refine `step_goal` to better match THIS step clip (based strictly on the {num_frames} frames).
  - Keep it as ONE coherent English sentence describing the intended intermediate world-state outcome of this step.
  - Use base/infinitive verb form consistently (e.g., 'Peel the outer skin off the onion' NOT 'Peeling...' or 'Peeled...').
  - Do NOT include actions or outcomes that are not supported by this step clip.
  - If the draft step_goal is overly broad, contains multiple independent actions, or includes incorrect details, rewrite it to be detailed and clip-consistent while staying coherent with the overall draft plan.
- Each `critical_frames[*].frame_index` MUST be an integer in [1, {num_frames}] and refers to the step-clip frame pool provided here.
- Choose 2 DISTINCT frames that show meaningful temporal progression within the step; do not pick duplicates.
- Keyframes MUST be chosen for their causal/visual significance within THIS step clip (do not pick frames solely because they are early/late).
- Your `critical_frames` MUST already be high-quality and image-aligned on the first pass; later alignment can only do minimal wording fixes and cannot change your chosen `frame_index`.
- In each `critical_frames[*]`, `causal_chain` MUST contain ONLY these 4 keys: `causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance` (and MUST NOT include `agent`/`action`/`patient`).
- In each `critical_frames[*]`, `interaction` MUST contain ONLY `patient`, `affordance_type`, and `mechanism` (do NOT output tools/materials and do NOT nest a `hotspot` object); `affordance_type` MUST be one lowercase token from the CANONICAL VOCABULARY below (use spaces, not underscores).

CANONICAL AFFORDANCE_TYPE VOCABULARY (use ONLY these tokens; choose the closest match):
  grasp point, cutting edge, pressing surface, contact surface, pouring lip, pivot point,
  support surface, sealing edge, handle, rim, lever arm, insertion point, friction surface,
  thermal surface, containment interior, opening, hinge, valve, screwing thread, gripping texture,
  impact surface, sliding surface, peeling point, tearing edge, rotation axis,
  clamping surface, flow channel, mixing surface, weight bearing surface,
  dispensing nozzle, knob, latch, drainage mesh, measuring mark.
  NOTE: `blade edge` is merged into `cutting edge` — always use `cutting edge` for any blade/edge used for cutting or slicing.
  If no token fits, use the MOST GENERAL applicable token from the list (e.g., `contact surface`).
- All required fields MUST be present and non-empty (no empty strings, empty arrays, empty objects, or null). In any string field (including list elements), do NOT reference frame/image indices, timestamps, durations, or timecodes. The only allowed frame reference is the integer `frame_index` field. Avoid placeholders like "N/A" or "unknown".

FORMAT STANDARD (applies to all `causal_*` list fields in this output, step-level and keyframe-level):
- Each `causal_*` field MUST be a JSON array of strings.
- Each string element MUST be a single, complete, objective English sentence grounded in the current step or key moment.
- Each string element MUST end with '.'.
- Each string element MUST NOT start with a list marker or numbering prefix (e.g., "1.", "2)", "-", "*", "•").
- Do NOT use newline characters inside any string element.
- MULTI-SENTENCE REQUIREMENT: each `causal_effect_on_*` list MUST contain at least 3 distinct sentences covering DIFFERENT aspects of the effect (e.g., one sentence for the patient's state change, one for the spatial rearrangement, one for the functional consequence). Do NOT write one long run-on sentence using "When..., resulting in..., thereby..." — break it into separate focused sentences.
- LIST SIZE: Step-level `causal_*` lists should contain 2–4 focused sentences each — enough to cover the key aspects without padding. Keyframe-level lists should contain 2–3 sentences each. Quality over quantity: every sentence must add meaningful information.
- ANTI-TEMPLATE: Do NOT begin every sentence with the same syntactic pattern (e.g., "When the...", "The..."). Vary sentence openings.
- NO CROSS-FIELD REPETITION: Do NOT copy phrases verbatim between fields (precondition, effect, rationale, step_goal, action_state_change_description). Each field must contribute UNIQUE information. If the same physical fact appears across fields, describe it from different perspectives using DISTINCT vocabulary.
- SPATIAL vs AFFORDANCE SEPARATION (CRITICAL — apply to ALL causal_* fields):
  `causal_*_on_spatial` fields describe POSITIONAL RELATIONSHIPS BETWEEN objects: where objects are relative to each other (contact, support, containment, above/below/beside, distance, orientation of one object relative to another). Ask: "WHERE is object A relative to object B?"
  `causal_*_on_affordance` fields describe INTRINSIC OBJECT STATES AND PROPERTIES: the object's own functional/mechanical state (open/closed, sealed/unsealed, empty/full, graspable due to surface texture, wet/dry, hot/cold, separated/clumped, locked/unlocked). Ask: "WHAT state is this object in? What can it do?"
  If a statement mixes both (e.g., "jar is on counter and lid is open"), SPLIT it: spatial → "jar is on counter", affordance → "jar lid is in open position."
  NEVER put intrinsic state changes (sealed→unsealed, open→closed, assembled→disassembled) into spatial fields.
  NEVER put positional/support/containment relationships into affordance fields.

TEMPORAL STRICTNESS (applies to ALL causal_precondition_* and causal_effect_* fields):
- Preconditions describe the world state BEFORE the action begins (one frame before any movement starts). They are enabling conditions, NOT mid-action descriptions.
- Effects describe the world state AFTER the action has FULLY COMPLETED (one frame after all movement stops). They are resulting states, NOT mid-action descriptions.
- Do NOT describe mid-action states as preconditions or effects. "Hand is gripping the handle" is mid-action if gripping IS part of this step. "Object is being lifted" is mid-action if lifting IS the action.

KEYFRAME-LEVEL TEMPORAL STRICTNESS (applies to ALL critical_frames[*].causal_chain fields):
- Same rule at keyframe level: preconditions = FROZEN state one frame BEFORE the micro-action begins; effects = FROZEN state one frame AFTER it completes. If you find yourself writing "hand is gripping" or "object is moving" as a precondition or effect, those are mid-action — rewrite.

ACTION-RELEVANCE FILTER: Every sentence MUST be directly causally related to the action being performed. Omit background objects, ambient scene details, and elements not involved in the operation.

Quality and grounding constraints:
- Treat the frames as the ONLY source of truth. Do not hallucinate objects, contacts, or states not supported by the images.
- Step-level `causal_chain.causal_precondition_on_*` and `causal_chain.causal_effect_on_*` MUST be MACRO summaries that integrate the entire step (not a single instant).
- Separation rule (IMPORTANT): Step-level `causal_chain.*` MUST stay MACRO and step-integrated, while keyframe-level `critical_frames[*].causal_chain.*` MUST be DETAILED and anchored to the specific keyframe image (more specific than the step-level chain; do NOT write a step-wide summary at the keyframe level).
- CROSS-STEP STATE INHERITANCE: This step's `causal_precondition_on_spatial` MUST be physically consistent with the PREVIOUS step's `causal_effect_on_spatial` as stated in the draft plan. If the previous step's effect says an object has been lifted or grasped, this step's precondition MUST NOT contradict that by claiming the object is still on its original surface.
- In each `critical_frames[*]`, `causal_chain.causal_precondition_on_spatial` and `causal_chain.causal_precondition_on_affordance` MUST describe the state of the world TRUE/REQUIRED AT that key moment, and MUST be visually consistent with the chosen image.
- In each `critical_frames[*]`, `causal_chain.causal_effect_on_spatial` and `causal_chain.causal_effect_on_affordance` MUST describe the PREDICTED immediate, local post-action effects right after the micro-action implied by `action_state_change_description` completes. "Immediate and local" means: within the same spatial locale and short timeframe as the micro-action itself. Do NOT write future-step preparation states, generalized final states, or workspace availability summaries (e.g., AVOID "will be ready for subsequent cooking", "countertop remains available", "tool is accessible for later use"). The phrase "predicted" means a short-term physical consequence that follows directly from the observed micro-action, NOT a distant future state.
- `interaction.patient/affordance_type/mechanism` must refer to a specific functional region that is visibly involved (edge, handle, rim, hinge, etc.) and explain a plausible physical mechanism matching the action type: friction for grasping, shear for cutting, gravity for placing/pouring, fluid drag for stirring, torque for rotating, thermal conduction/convection/radiation for cooking. Do NOT default to "friction" for all actions.
- Use consistent object naming across all fields; do not rename the same object with different synonyms within the step.
- CROSS-STEP NAMING CONSISTENCY (HARD RULE): Object names in this step MUST match the names used in the draft plan outline above. If the draft plan uses "black wok" in Step 2, every step that mentions that wok MUST also use "black wok" — not "wok", "dark wok", "cast iron wok", or "large pan". This applies to `patient`, `agent`, `step_goal`, `rationale`, and all `causal_*` sentences. Refer to the draft plan outline's entity names as the authoritative registry.
- Prefer concrete, mechanistic relations and state terms (e.g., contacting, holding, inside, aligned_with, open/closed) rather than vague language.

SPATIAL AND AFFORDANCE ANNOTATION GUIDELINES:
(Apply the SPATIAL vs AFFORDANCE SEPARATION rule defined above — spatial = positional relationships between objects; affordance = intrinsic object states/properties.)

SPATIAL: Each sentence must name two entities and their visual spatial relationship (position, contact, containment, support). Avoid abstract terms like "accessible/within reach."

AFFORDANCE: Only include affordances visible or implied by mechanical state. Do NOT assert hidden qualities or vague readiness ("ready for X", "available for Y"). Focus on: mechanical state (open/closed, sealed/unsealed), container state (empty/full), manipulation-enabling properties (graspable due to texture, pourable with unobstructed opening). When describing affordances, include the physical property that enables them.

OBSERVABILITY: Only assert directly observable properties (position, contact, color, shape, open/closed). Do NOT assert invisible internal properties (temperature, chemical composition, structural fatigue) unless visually confirmed (e.g., visible steam = hot). No DO/NDO labels in output.

AFFORDANCE EFFECT HIERARCHY: `causal_effect_on_affordance` must prioritize: (1) patient's state change first, (2) tool state, (3) environment only if essential. No "ready for X" standalone effects.

KEYFRAME EFFECTS: `critical_frames[*].causal_effect_on_*` must be IMMEDIATE and LOCAL — the direct consequence of the micro-action. No future-step states or workspace availability summaries.

CAUSAL CHAIN COMPLETENESS: Each keyframe's annotation must cover: (1) spatial setup, (2) affordance mechanism, (3) force/action, (4) concrete result on the patient. The RESULT is most commonly omitted — always verify it's present.

Examples (follow the GOOD style):
- Bad (keyframe effect): "The countertop will remain available for subsequent preparation."
  Good (keyframe effect): [
    "Onion outer layer is partially detached from flesh at the point of knife contact.",
    "Knife blade has penetrated through the dry outer skin layer."
  ]

Output schema (strict):

Field guide (read carefully; semantic, not formatting):
- `step_id` (int): Must equal the draft `step_id` exactly (read-only).
- `step_goal` (string): Refine the draft `step_goal` into ONE detailed English sentence that matches THIS step clip. Use base/infinitive verb form (e.g., 'Peel...' not 'Peeling...').
- `rationale` (string): One natural, accurate English sentence explaining how this step contributes to the high_level_goal — why it is necessary for the overall plan to succeed. Do NOT just restate `step_goal`. Do NOT use generic justifications like 'improves hygiene', 'ensures safety', 'ensures proper preparation', or 'maintains cleanliness'. Focus on the specific physical role this step plays in achieving the video's overall objective — explain what would be incomplete or impossible in the plan without this step.
- `causal_chain` (object): Step-level MACRO physical causal analysis for the ENTIRE step:
  - `agent` (string): Primary force/controller for the whole step (prefer body part like 'hands'/'left_hand'/'right_hand'; use a tool part only if it is clearly the direct force applicator). Use one stable identifier.
  - `action` (string): Physical verb phrase for the whole step (include mechanism when possible: push/pull/rotate/tilt/insert/press). BANNED VAGUE VERBS: do, use, handle, manipulate, interact with, work on, manage, deal with, process, operate, arrange, organize, prepare, set up, move (when used alone without direction). Use specific physical verbs instead: push, pull, rotate, tilt, insert, press, grasp, lift, lower, place, release, slide, pour, cut, peel, tear, fold, unfold, screw, unscrew, wipe, rinse, squeeze, stir, shake, flip, tap, align, withdraw, stabilize, adjust, carry, transport.
  - `patient` (string): Primary acted-on object identifier (use spaces between words, e.g. 'dirty plate', 'rice cooker pot'). Keep naming consistent across all fields (do not rename the same object).
  - `causal_precondition_on_spatial` (list[str]): MACRO spatial preconditions for the ENTIRE step — describe POSITIONAL RELATIONSHIPS BETWEEN objects that MUST ALREADY HOLD BEFORE this step begins: contact, support, containment, relative position (inside, on_top_of, beside, above). Include only ESSENTIAL spatial preconditions; omit incidental scene layout details. Do NOT include intrinsic object states (open/closed, sealed/unsealed) — those belong in affordance. Use FORMAT STANDARD. (TEMPORAL STRICTNESS: BEFORE ANY action begins.)
  - `causal_precondition_on_affordance` (list[str]): MACRO affordance preconditions for the ENTIRE step — describe INTRINSIC OBJECT STATES AND PROPERTIES that MUST ALREADY HOLD BEFORE this step begins: functional/mechanical state (sealed/unsealed, open/closed, empty/full, locked/unlocked), surface properties enabling manipulation (dry/textured for grip, sharp edge for cutting). MUST be DISTINCT FROM spatial preconditions: focus on what the object IS, not where it is. SPECIFICITY REQUIREMENT: BANNED standalone terms — do NOT write just 'graspable', 'pourable', 'cuttable' alone. ALWAYS state the PHYSICAL PROPERTY (e.g., 'handle has textured rubber coating providing non-slip grip' NOT just 'handle is graspable'). Use FORMAT STANDARD. (TEMPORAL STRICTNESS: BEFORE ANY action begins.)
  - `causal_effect_on_spatial` (list[str]): MACRO spatial effects AFTER the ENTIRE step completes — describe how POSITIONAL RELATIONSHIPS BETWEEN objects changed. STATE-CHANGE LANGUAGE: use transition markers (e.g., 'Onion is now inside the pan (was on the cutting board)', 'Knife has moved from counter to drying rack'). Focus on WHERE objects moved, what new contact/support/containment relationships hold. Do NOT include intrinsic state changes (sealed→unsealed, open→closed) — those belong in affordance effects. Use FORMAT STANDARD. (TEMPORAL STRICTNESS: AFTER ALL action completes.)
  - `causal_effect_on_affordance` (list[str]): MACRO affordance effects AFTER the ENTIRE step completes — describe how INTRINSIC OBJECT STATES changed. MUST follow AFFORDANCE EFFECT HIERARCHY: start with the patient's core functional state change (e.g., 'seal is broken, exposing contents', 'lid is now in open position'), then tool state, then environment only if essential. Do NOT include positional changes — those belong in spatial effects. Do NOT write only workspace/surface availability. Do NOT use 'ready for X / available for Y / accessible for Z' as primary effects. Use FORMAT STANDARD. (TEMPORAL STRICTNESS: AFTER ALL action completes.)
- `counterfactual_challenge_question` (string): One realistic counterfactual what-if question that could disrupt this step due to physics/constraints, grounded in the scene. MUST start with 'What if ...?'. The what-if MUST target a SPECIFIC physical/spatial/affordance condition involving a VISIBLE object or relation in the current scene (not a vague "what if it was harder/slower"). This field is ONLY about a counterfactual disruption; do NOT mix in non-counterfactual failure analysis.
- `expected_challenge_outcome` (string): Predicted physical outcome if that counterfactual challenge occurs. MUST be ONE single, specific, immediate physical consequence grounded in this step's spatial setup and affordances. SECOND-ORDER REASONING: after stating the immediate consequence, explain what downstream task outcome this would prevent or alter. Do NOT stack multiple independent cascading consequences. Do NOT propose any recovery actions, alternative tools, or workarounds. Do NOT write generic safety/hygiene/delay outcomes. Instead describe the CONCRETE physical result on the patient/task and its downstream impact.
- `failure_reflecting` (object): Real (non-counterfactual) failure analysis for this step:
  - `reason` (string): Most plausible real failure mode. SEVERITY REQUIREMENT: must substantially block or derail step completion (not just reduce efficiency or slightly degrade quality). GROUNDING REQUIREMENT: mechanism must be based on visible physical conditions, not invisible or speculative causes. SPECIFICITY REQUIREMENT: MUST name the SPECIFIC object(s) and the SPECIFIC physical mechanism — do NOT use generic language like 'a bulky item', 'an object' when you can name the actual patient/agent. THEMATIC COHERENCE (STRONGLY PREFERRED): should target the SAME physical vulnerability domain as the counterfactual_challenge_question. However, if the most plausible real failure is in a different domain, write the most plausible failure instead of forcing a weak thematic match.
  - `recovery_strategy` (string): ONE concrete, physically plausible recovery action (a single key maneuver, not a multi-step script). SAFETY: must be safe and hygienic (do not retrieve food from floor/drain). MINIMAL: only restore the specific broken condition; do not rewrite the entire step. SPECIFICITY: name the specific object and describe the maneuver in enough detail to be actionable (e.g., 'Rotate the wok handle downward to clear the rail' NOT just 'Adjust the position'). ANTI-PARROT: recovery MUST use DIFFERENT vocabulary and framing than the failure reason. Do not introduce new unseen tools/objects.
- `critical_frames` (list): MUST contain exactly 2 objects. These are the two most causally important and visually anchorable key moments within the step (NOT limited to initiation/completion).
  Each `critical_frames[*]` object contains:
  - `frame_index` (int): 1-based index into THIS step-clip frame pool (1..{num_frames}); the 2 indices must be distinct and strictly increasing.
  - `action_state_change_description` (string): Describe BOTH the action happening at this key moment AND the specific state change it causes. You MUST include: (1) the action being performed (who does what to whom), AND (2) the explicit BEFORE→AFTER state transition — name the property that changes and its state before and after (e.g., 'Right hand closes around the pot handle, transitioning from open-palm hovering to closed-grip contact — establishing finger-to-handle friction that supports the pot's weight' NOT just 'Person grabs the pot'). Be specific and grounded in the image: name the actor, patient, contact points. For pick-and-place actions, describe the contact or support change (e.g., 'pot base contact transfers from stove surface to hand support'). Do NOT write only a static pose description. Do NOT write only an action caption without the state change — every description MUST contain both ACTION and STATE CHANGE components.
  - `causal_chain` (object): Keyframe-level causal analysis with EXACTLY these 4 fields (no agent/action/patient). Each field uses FORMAT STANDARD:
    - `causal_precondition_on_spatial` (list[str]): DETAILED positional relationships BETWEEN objects FROZEN one frame BEFORE the micro-action begins — describe WHERE objects are relative to each other (contact, support, containment, relative position). Must be visually consistent with the chosen image. Do NOT include intrinsic object states here. NOT mid-action. (TEMPORAL STRICTNESS: BEFORE the action.)
    - `causal_precondition_on_affordance` (list[str]): DETAILED intrinsic object states/properties FROZEN one frame BEFORE the micro-action begins — describe WHAT functional/mechanical state each relevant object is in (open/closed, sealed/unsealed, graspable due to surface texture). Do NOT include positional relationships here. Must be visually consistent with the chosen image. NOT mid-action. (TEMPORAL STRICTNESS: BEFORE the action.)
    - `causal_effect_on_spatial` (list[str]): PREDICTED changes to positional relationships BETWEEN objects one frame AFTER the micro-action completes — how objects moved relative to each other (new contact, support gained/lost, containment change). Do NOT include intrinsic state changes here. MUST be immediate and local. NOT mid-action. (TEMPORAL STRICTNESS: AFTER the action.)
    - `causal_effect_on_affordance` (list[str]): PREDICTED changes to intrinsic object states one frame AFTER the micro-action completes — how the object's own functional/mechanical state changed (seal broken, lid opened, grip established). Do NOT include positional changes here. MUST follow AFFORDANCE EFFECT HIERARCHY (patient state first). MUST be immediate and local. NO 'ready for X' / 'available for Y'. NOT mid-action. (TEMPORAL STRICTNESS: AFTER the action.)
	  - `interaction` (object): MUST contain ONLY these 3 keys:
	    - `patient` (string): Specific functional region of the patient object involved (e.g., handle, rim, edge, hinge); keep it concrete and visually grounded.
	    - `affordance_type` (string): One lowercase token from the CANONICAL VOCABULARY (grasp point, cutting edge, pressing surface, contact surface, pouring lip, pivot point, support surface, sealing edge, handle, rim, lever arm, insertion point, friction surface, thermal surface, containment interior, opening, hinge, valve, screwing thread, gripping texture, impact surface, sliding surface, peeling point, tearing edge, rotation axis, clamping surface, flow channel, mixing surface, weight bearing surface, dispensing nozzle, knob, latch, drainage mesh, measuring mark). NOTE: `blade edge` is merged into `cutting edge`. Choose the closest match.
	    - `mechanism` (string): Physical mechanism describing how interaction at this region achieves the micro-action, grounded in what is visible. ACTION-MECHANISM MATCHING (do NOT default to "friction" for all actions):
	      - Grasping/holding: friction + normal force between fingers and surface texture
	      - Cutting/slicing: shear force from blade edge penetrating material
	      - Placing/releasing: gravity acting on the object after support withdrawal
	      - Pouring/tilting: gravity-driven fluid/granular flow through opening
	      - Stirring/mixing: fluid drag and viscous shear from implement motion through liquid/semi-solid
	      - Pressing/pushing: normal force application through rigid contact
	      - Rotating/twisting: torque around axis of rotation
	      - Cooking/heating: conduction (pan→food), convection (boiling liquid→food), radiation (flame→food)
	      - Peeling/tearing: tensile force separating bonded layers
	      - Opening/unscrewing: torque applied through grip on threaded or hinged closure
	      - Washing/rinsing: water flow (gravity or pressure) carrying away surface contaminants
	      - Folding/wrapping: bending force converting flat material into layered configuration
	      - Squeezing/expressing: compressive force expelling contents through opening or pores
	      - Scooping/ladling: implement motion through granular/fluid medium, gravity retaining contents in concave surface
	      - Wiping/scrubbing: lateral friction force between cleaning surface and target surface
	      Choose the mechanism that matches the ACTUAL physical action, not a generic fallback.

Output JSON template (keep keys exactly):
{{
  "step_id": 1,
  "step_goal": "Refine the draft step_goal into ONE detailed English sentence that matches THIS step clip.",
  "rationale": "One natural, accurate English sentence explaining how this step contributes to the high_level_goal — why it is necessary for the overall plan to succeed. Do NOT just restate step_goal. No generic justifications. Explain what would be incomplete without this step.",
  "causal_chain": {{
    "agent": "Primary force/controller for the whole step (prefer body part like 'hands'/'left_hand'/'right_hand'; use a tool part only if it is clearly the direct force applicator). Use one stable identifier.",
    "action": "Physical verb phrase for the whole step (include mechanism when possible: push/pull/rotate/tilt/insert/press). BANNED VAGUE VERBS: do, use, handle, manipulate, interact with, work on, manage, deal with, process, operate, arrange, organize, prepare, set up, move (when used alone without direction). Use specific physical verbs instead: push, pull, rotate, tilt, insert, press, grasp, lift, lower, place, release, slide, pour, cut, peel, tear, fold, unfold, screw, unscrew, wipe, rinse, squeeze, stir, shake, flip, tap, align, withdraw, stabilize, adjust, carry, transport.",
    "patient": "Primary acted-on object identifier (use spaces between words). Keep naming consistent across all fields (do not rename the same object).",
    "causal_precondition_on_spatial": ["Positional relationships between objects BEFORE step begins."],
    "causal_precondition_on_affordance": ["Intrinsic object states/properties BEFORE step begins."],
    "causal_effect_on_spatial": ["How positional relationships changed AFTER step completes."],
    "causal_effect_on_affordance": ["How intrinsic object states changed AFTER step completes — patient first."]
  }},
  "counterfactual_challenge_question": "One realistic counterfactual what-if question targeting a SPECIFIC visible physical/spatial/affordance condition. MUST start with 'What if ...?'. Counterfactual disruption ONLY; do NOT mix in non-counterfactual failure analysis.",
  "expected_challenge_outcome": "ONE single, specific, immediate physical consequence. No recovery actions, no cascading consequences, no generic safety/delay outcomes.",
  "failure_reflecting": {{
    "reason": "Most plausible real failure mode that would SUBSTANTIALLY BLOCK step completion (not mild inefficiency). Grounded in visible physical mechanism.",
    "recovery_strategy": "ONE concrete recovery maneuver (not a multi-step script). Safe, hygienic, minimal — only restore the broken condition. No unseen tools."
  }},
  "critical_frames": [
    {{
      "frame_index": 1,
      "action_state_change_description": "Key moment 1 (earlier than Key moment 2): Describe BOTH the action AND the BEFORE→AFTER state change. Name the actor, patient, contact points. State the property that changes and its before/after states. Do NOT write only an action caption — MUST include explicit state transition.",
      "causal_chain": {{
        "causal_precondition_on_spatial": ["Positional relationships between objects BEFORE this micro-action."],
        "causal_precondition_on_affordance": ["Intrinsic object states BEFORE this micro-action."],
        "causal_effect_on_spatial": ["How positional relationships changed AFTER this micro-action."],
        "causal_effect_on_affordance": ["How intrinsic object states changed AFTER this micro-action — patient first."]
      }},
      "interaction": {{
        "patient": "Specific functional region involved (e.g., handle, rim, edge, hinge).",
        "affordance_type": "grasp point",
        "mechanism": "Explain the physical mechanism grounded in what is visible."
      }}
    }},
    {{
      "frame_index": 2,
      "action_state_change_description": "Key moment 2 (later than Key moment 1): Describe BOTH the action AND the BEFORE→AFTER state change. Name the actor, patient, contact points. State the property that changes and its before/after states. Do NOT write only an action caption — MUST include explicit state transition.",
      "causal_chain": {{
        "causal_precondition_on_spatial": ["Positional relationships between objects BEFORE this micro-action."],
        "causal_precondition_on_affordance": ["Intrinsic object states BEFORE this micro-action."],
        "causal_effect_on_spatial": ["How positional relationships changed AFTER this micro-action."],
        "causal_effect_on_affordance": ["How intrinsic object states changed AFTER this micro-action — patient first."]
      }},
      "interaction": {{
        "patient": "Specific functional region involved (edge, handle, rim, hinge, etc.).",
        "affordance_type": "contact surface",
        "mechanism": "Explain the physical mechanism grounded in what is visible."
      }}
    }}
  ]
}}

High-level goal (context): {high_level_goal}

Draft plan outline (for coherence across steps; you may refine ONLY the current step_goal):
{draft_plan_outline}

Reference draft step JSON (read-only; do not echo it in output):
```json
{draft_step_json}
```
""".strip()

def build_stage3_keyframe_alignment_user_prompt(
    *,
    step_id: int,
    step_goal: str,
    critical_frames_json: str,
    num_frames: int,
    frame_indices_1based: list[int],
) -> str:
    indices = ", ".join(str(int(x)) for x in frame_indices_1based)
    return f"""
You are an expert Physical Interaction Analyst and Causal Planner.
You are given TWO selected keyframe images from a SINGLE STEP CLIP with {num_frames} uniformly sampled frames (chronological order).
The two images correspond to these locked 1-based indices in the FULL step-clip frame pool: [{indices}].

Task:
Make the keyframe annotations EXACTLY match the provided images.
You are fixing alignment issues where the saved keyframe images and the JSON `critical_frames` descriptions can drift.

Strict requirements:
- You MUST NOT change `step_id` (read-only) or `step_goal` (read-only).
- You MUST NOT change the provided `frame_index` values; they are LOCKED to the images you see.
- You MUST output ONLY one JSON object with a single top-level key `critical_frames` (no other top-level keys).

For each `critical_frames[*]` object:
- `frame_index` (int): Must equal one of the locked indices exactly.
- `action_state_change_description` (string): Describe BOTH the action visible in this image AND the specific BEFORE→AFTER state change it causes. You MUST include: (1) the action being performed, AND (2) the explicit state transition — name the property that changes and its before/after states (e.g., 'gripped→released', 'contact gained', 'support transferred'). Must be directly verifiable in the image. For pick-and-place, describe the contact/support change. Do NOT write only a static pose. Do NOT write only an action caption without the state change.
- `causal_chain` (object): MUST contain ONLY these 4 keys:
  `causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`.
- `interaction` (object): MUST contain ONLY `patient`, `affordance_type`, and `mechanism`.
  - `affordance_type` MUST be one lowercase token from the CANONICAL VOCABULARY: grasp point, cutting edge, pressing surface, contact surface, pouring lip, pivot point, support surface, sealing edge, handle, rim, lever arm, insertion point, friction surface, thermal surface, containment interior, opening, hinge, valve, screwing thread, gripping texture, impact surface, sliding surface, peeling point, tearing edge, rotation axis, clamping surface, flow channel, mixing surface, weight bearing surface, dispensing nozzle, knob, latch, drainage mesh, measuring mark. NOTE: `blade edge` is merged into `cutting edge`. Choose the closest match.

FORMAT STANDARD (applies to all `causal_*` list fields):
- Each `causal_*` field MUST be a JSON array of strings.
- Each string element MUST be a single, complete, objective English sentence grounded in the image.
- Each string element MUST end with '.'.
- Each string element MUST NOT start with a list marker or numbering prefix (e.g., "1.", "2)", "-", "*", "•").
- Do NOT use newline characters inside any string element.

SPATIAL AND AFFORDANCE ANNOTATION GUIDELINES:

KEY RULES (apply from system prompt and Stage 3):
- SPATIAL vs AFFORDANCE: spatial fields = positional relationships between objects; affordance fields = intrinsic object states/properties. Never mix them.
- TEMPORAL STRICTNESS: preconditions = state BEFORE the micro-action; effects = state AFTER it completes. No mid-action descriptions.
- ACTION-RELEVANCE: only include objects/states causally related to the action. Omit background details.
- OBSERVABILITY: only assert directly observable properties. No invisible internal states unless visually confirmed.
- AFFORDANCE EFFECT HIERARCHY: patient state change first, then tool, then environment. No "ready for X" language.
- Keyframe effects must be IMMEDIATE and LOCAL — no future-step states or workspace summaries.

Examples (contrast; follow the GOOD style):
SPATIAL examples:
- Bad: "The spatula is within reach."
  Good: [
    "Spatula_handle is in contact with right_hand.",
    "Spatula_head is above pan_interior."
  ]
AFFORDANCE examples (intrinsic object states/properties — NOT positional relationships):
- Bad: "The burner is functional."
  Good: [
    "Burner element is in heated state (visible glow or steam indicating active thermal output).",
    "Burner control knob is in the 'on' position (heat is being generated)."
  ]

Global bans:
- In any free-form text field, do NOT reference frame/image indices or timestamps/durations/timecodes.
- Return JSON only: no markdown, no comments, no extra text.

Minimal-edit preference:
- Make the SMALLEST edits needed to fix mismatches.
- If an existing field is already correct for the image, keep it unchanged.

Context (read-only):
- step_id: {int(step_id)}
- step_goal: {step_goal}

Existing critical_frames (for reference; fix any mismatches, but keep frame_index locked):
```json
{critical_frames_json}
```

Output JSON template (keep keys exactly):
{{
  "critical_frames": [
    {{
      "frame_index": {int(frame_indices_1based[0]) if frame_indices_1based else 1},
      "action_state_change_description": "Describe the action AND the before→after state change visible at this keyframe.",
      "causal_chain": {{
        "causal_precondition_on_spatial": ["Positional relationships between objects BEFORE this micro-action."],
        "causal_precondition_on_affordance": ["Intrinsic object states BEFORE this micro-action."],
        "causal_effect_on_spatial": ["How positional relationships changed AFTER this micro-action."],
        "causal_effect_on_affordance": ["How intrinsic object states changed AFTER this micro-action — patient first."]
      }},
      "interaction": {{
        "patient": "Specific functional region (handle, rim, edge, hinge).",
        "affordance_type": "grasp point",
        "mechanism": "Physical mechanism grounded in visible evidence."
      }}
    }},
    {{
      "frame_index": {int(frame_indices_1based[1]) if len(frame_indices_1based) > 1 else 2},
      "action_state_change_description": "Describe the action AND the before→after state change visible at this keyframe.",
      "causal_chain": {{
        "causal_precondition_on_spatial": ["Positional relationships between objects BEFORE this micro-action."],
        "causal_precondition_on_affordance": ["Intrinsic object states BEFORE this micro-action."],
        "causal_effect_on_spatial": ["How positional relationships changed AFTER this micro-action."],
        "causal_effect_on_affordance": ["How intrinsic object states changed AFTER this micro-action — patient first."]
      }},
      "interaction": {{
        "patient": "Specific functional region (edge, handle, rim, hinge).",
        "affordance_type": "contact surface",
        "mechanism": "Physical mechanism grounded in visible evidence."
      }}
    }}
  ]
}}
""".strip()

def build_stage3_hlg_and_detail_independence_prompt(
    *,
    draft_high_level_goal: str,
    draft_plan_outline: str,
    refined_plan_outline: str,
    refined_steps_json: str,
) -> str:

    return f"""
You are an expert Physical Interaction Analyst and Causal Planner.

You are given sampled frames from the FULL original video in chronological order as visual evidence, along with the draft plan, refined step outlines, and refined step annotations that include `independence` labels.

You must produce a SINGLE JSON object that contains:
1. A refined `high_level_goal` for the entire video.
2. A `detail_independence` explanation for each step (except Step 1).

--- PART A: Refine high_level_goal ---

Refine the overall `high_level_goal` into ONE comprehensive English sentence describing the overall goal and intended final outcome of the ENTIRE video.
This refinement happens AFTER all step-level annotations are generated; it MUST be consistent with the refined step goals.

Rules for high_level_goal:
- `high_level_goal` MUST be one English sentence that captures ALL major activity phases visible in the video from start to finish — do not drop early, intermediate, or preparatory phases. If the video has a preparatory phase that enables a main phase, describe both using a subordinating structure (e.g., 'After clearing the workspace, prepare X and serve it at Y'). Do NOT list every step, but DO mention each distinct PURPOSE.
- ENUMERATION SELF-CHECK (MANDATORY): Cross-reference the refined step goals outline below — every step's distinct purpose MUST be reflected in the high_level_goal. If any step's purpose is missing from the high_level_goal, the goal is incomplete and MUST be rewritten. The high_level_goal must serve as a complete summary that a reader can use to understand ALL major events in the video without reading individual steps.
- `high_level_goal` MUST NOT reference frame/image indices, timestamps, durations, or timecodes.
- Avoid placeholders like "unknown", "N/A", "...".
- Prefer a minimal edit if the draft high_level_goal is already correct, but fix any incorrect/broad details so it matches the refined steps.

Draft high_level_goal (context; may be imperfect):
{draft_high_level_goal}

Draft plan outline (context):
{draft_plan_outline}

Refined step goals outline (authoritative for this refinement):
{refined_plan_outline}

--- PART B: Produce detail_independence ---

For each step except Step 1, produce `detail_independence` based on the step's `independence` value in the refined annotations below.

Rules for detail_independence:
- If `independence` is `"yes"`, write one grounded English sentence explaining how the previous step's visible effect enables a required precondition of the current step.
  QUALITY STANDARD for independence="yes":
  (a) MUST name the SPECIFIC physical effect from the previous step (e.g., "the wok is stored in the lower cabinet" not "the previous step completed").
  (b) MUST explain the PHYSICAL MECHANISM of dependency — WHY the current step cannot proceed without that effect (e.g., "which frees space on the drying rack for the steamer insert" not just "which is necessary for this step").
  (c) MUST reference a VISIBLE, OBSERVABLE state change — not abstract readiness (e.g., "the mug visibly contains dry cereal below the rim" not "the mug is prepared").
  (d) MUST NOT use vague dependency language: "is necessary for", "is required for", "is needed for", "enables" as standalone justifications. Always state WHY.
  Examples:
  - GOOD: "After the wok is stored in the lower cabinet, the drying rack is visibly freed of its largest item, which opens space for repositioning the cutting board and steamer insert."
  - GOOD: "The previous step leaves the white mug upright on the cleared counter with the cabinet open, enabling cereal to be taken out and poured directly into the mug."
  - BAD: "The previous step is necessary for this step to proceed." (no specifics)
  - BAD: "Step 2's effects enable Step 3." (no physical mechanism)
  - BAD: "The mug is prepared for cereal." (vague readiness, no observable state)
- If `independence` is `"no"`, set `detail_independence` to the empty string `""`.
- Use the sampled frames and the refined step causal chains as evidence.
- Be factual and conservative. Do not invent hidden object states, intentions, or off-screen events.
- Do not reference frame indices, timestamps, or durations.

Refined steps with causal context:
```json
{refined_steps_json}
```

--- OUTPUT ---

Output MUST be a single, syntactically valid JSON object with exactly two top-level keys: `high_level_goal` and `steps`.
- `high_level_goal` (string): The refined sentence.
- `steps` (list): One entry per step_id from 2 through the last step, each containing exactly `step_id` (int) and `detail_independence` (string).

First determine the refined high_level_goal, then produce detail_independence for each step consistent with it.

Output JSON template (keep keys exactly):
{{
  "high_level_goal": "One comprehensive English sentence describing the overall goal and intended final outcome of the entire video.",
  "steps": [
    {{
      "step_id": 2,
      "detail_independence": "One grounded sentence explaining how the previous step enables this step, or an empty string if independence is no."
    }},
    {{
      "step_id": 3,
      "detail_independence": ""
    }}
  ]
}}
""".strip()


def build_stage4_user_prompt(
    high_level_goal: str,
    step_goal: str,
    step_annotation_json: str,
    num_frames: int,
    next_step_goal: str = "",
    global_entity_registry: str = "",
) -> str:

    next_step_line = ""
    if next_step_goal:
        next_step_line = f"\nNEXT step goal (read-only; for boundary enforcement ONLY — do NOT generate actions for this): {next_step_goal}"
    entity_registry_block = ""
    if global_entity_registry:
        entity_registry_block = f"""
GLOBAL ENTITY REGISTRY (MANDATORY — cross-step naming consistency):
The following canonical object names are used across ALL steps of this video. When any of these objects appears as `patient` in your atomic actions, you MUST use the EXACT string from this registry. Do NOT invent synonyms, abbreviations, or alternative names.
{global_entity_registry}
"""
    return f"""
You are an expert Physical Interaction Analyst specializing in fine-grained atomic action decomposition.
You are given {num_frames} uniformly sampled frames from a SINGLE STEP CLIP (chronological order).

DARK/CORRUPTED FRAME GUARD: If the majority of these step-clip frames are entirely dark, black, heavily occluded, or show no discernible activity, output a JSON object with `"error": "no_visual_content"` and `"reason": "Majority of step-clip frames are dark/black/corrupted with no visible activity."` instead of generating atomic actions. Do NOT hallucinate actions or objects from featureless frames.

High-level goal (context): {high_level_goal}
Step goal (THIS step): {step_goal}{next_step_line}
{entity_registry_block}
Reference step annotation (read-only; for context — do NOT echo in output):
```json
{step_annotation_json}
```

Task:
Decompose this step into ATOMIC ACTIONS — complete, self-contained physical operations that each accomplish one clear functional sub-goal.
An atomic action captures a COMPLETE INTENT-TO-OUTCOME cycle: from the moment the actor begins a goal-directed motion to the moment that sub-goal is achieved (object grasped, object placed, cut completed, door opened).
Do NOT decompose into kinematic primitives (reach, grasp, lift, carry as separate entries). Instead, group the full motion chain that serves one functional goal into ONE atomic action.
Think: "What would a human annotator label as ONE operation when watching at normal speed?"

RECOMMENDED PROCEDURE (follow silently):
1) Scan ALL frames first to understand the full motion trajectory and state changes within this step.
2) Identify the critical state-change boundaries: moments where the agent-patient contact changes (contact established / broken), motion direction reverses, a new object is engaged, or a distinct sub-goal is achieved.
3) Use the reference step annotation's `critical_frames` (if present) as ANCHOR POINTS: boundaries of atomic actions should generally align with or bracket these key moments. If the two critical frames suggest a state change between frame_index A and B, place an atomic action boundary near that transition.
4) For each segment between boundaries, assign exactly one atomic action with the correct verb and patient.
5) Perform an explicit self-check: for every boundary frame, verify that the frame visually shows the claimed transition (e.g., contact just established, object just lifted, hand just released). If a mismatch exists, adjust the boundary by ±1 frame.

For each atomic action, predict:
- `atomic_action_id`: sequential 1-based integer
- `start_frame_index`: 1-based inclusive start boundary on the {num_frames}-frame pool
- `end_frame_index`: 1-based exclusive end boundary (half-open `[start, end)`)
- `actor`: the specific body part or tool part that is the direct force applicator (use lowercase with spaces; prefer specifics like `right hand`, `left hand`, `both hands`, `right thumb and index`, `knife blade`, `spatula head`, `pliers jaw`; avoid vague `hand` or `person`)
- `action`: a concrete verb phrase describing the atomic physical action including the physical mechanism (e.g., "reach toward the cup handle with fingers extended", "apply pinch grip on the cup handle", "lift the cup vertically 10cm off the counter"); avoid vague verbs like "do", "use", "interact with", "handle". NATURAL LANGUAGE: use natural English in this field (e.g., "the rice cooker pot", "the dirty plate") — do NOT use underscores in prose
- `patient`: the primary object being acted upon (lowercase with spaces, e.g. 'dirty plate', 'rice cooker pot'; consistent naming with the step annotation; must be exactly ONE entity — if multiple objects, name the primary and mention secondary objects in caption)
- `caption`: one detailed English sentence describing the atomic action, which MUST include: (a) the spatial relationship between actor and patient at the START of this action, (b) the physical motion or force application, (c) the visible state change by the END of this action. GROUNDING RULE: all three components (a), (b), (c) must be directly verifiable in the frames. Do NOT describe hidden internal states (e.g., "applying even pressure throughout the meat"), inferred material properties (e.g., "the sharp blade cuts through"), or future-step readiness (e.g., "preparing for subsequent cooking"). Describe ONLY what is visually observable: positions, contacts, motions, orientations, and visible state changes. NATURAL LANGUAGE: use natural English everywhere — do NOT use underscores in any text field. Example: "The right hand, positioned above the cup handle, descends to wrap fingers around the ceramic handle, establishing a pinch grip with thumb on top and three fingers curled underneath."
  CAPTION SELF-CHECK: After writing each caption, verify that it describes: (a) the spatial arrangement at the start, (b) the physical motion, and (c) the resulting state or spatial change by the end. State changes should be described naturally — explicit "from X to Y" phrasing is helpful but not mandatory. For transition/carry actions where no object property changes, describe the spatial progression (e.g., "from the sink area to the cabinet area").

FRAME BOUNDARY CONSTRAINTS (NON-NEGOTIABLE):
- `start_frame_index` of the FIRST atomic action MUST be `1`.
- `end_frame_index` of the LAST atomic action MUST be `{num_frames + 1}` (exclusive boundary after the last frame; ensures full coverage).
- Contiguous: for consecutive atomic actions, `end_i == start_(i+1)` (no gaps, no overlaps).
- Each atomic action MUST satisfy `start_frame_index < end_frame_index`.
- All indices MUST be integers in [1, {num_frames + 1}]; `{num_frames + 1}` is allowed ONLY for the last `end_frame_index`.

BOUNDARY ACCURACY (NON-NEGOTIABLE):
- Place boundaries at the frame where a VISIBLE STATE CHANGE occurs: contact established/broken, motion direction change, new object engagement, sub-goal completion.
- For each boundary, you MUST be able to point to a specific visual difference between the frame before and after.
- Do NOT place boundaries at arbitrary equal intervals; follow the actual motion dynamics.
- When uncertain between two adjacent frames, choose the frame where the state change is more clearly visible.

DECOMPOSITION GUIDELINES — SEMANTIC COMPLETENESS:
- CORE PRINCIPLE: Each atomic action represents ONE COMPLETE FUNCTIONAL OPERATION. Split at changes in INTENT (different object, different goal), NOT at changes in hand kinematics.
  WHAT IS ONE OPERATION (keep as a single atomic action):
  - The full pick-up cycle: approach + grasp + lift = ONE action "pick up [object]"
  - The full put-down cycle: carry + lower + place + release = ONE action "place [object] on [surface]"
  - The full open/close cycle: grasp handle + pull/push door = ONE action "open/close [object]"
  - A complete cut: position knife + press/saw through = ONE action "cut [object]"
  - A pour: tilt container + liquid flows + right container = ONE action "pour [substance] into [target]"
  - A wipe/spread: tool contacts surface + sweeps across = ONE action "spread [substance] across [surface]"
  - A button/knob interaction: reach + press/turn = ONE action "press [button]" or "turn [knob]"
  - A scoop-and-deposit cycle: dip into source + load + carry to target + deposit = ONE action "scoop [substance] from [source] onto [target]"
  - A tool-assisted transfer: fork/tongs into container + lift food + carry + deposit = ONE action "transfer [food] from [source] to [target]"
  WHAT REQUIRES SPLITTING (separate atomic actions):
  - Actor switches to a DIFFERENT object (after placing knife, picks up fork → two actions)
  - A clearly different functional goal begins (after stirring, starts scooping → two actions)
  - A significant pause or direction reversal separates two sub-goals
  - The actor's hands switch roles (left hand takes over from right hand)
  REPEATED CYCLES: When the same gesture repeats in a loop (chop-chop-chop, tear+place+tear+place, stir-stir-stir), treat the ENTIRE repetition as ONE atomic action unless the patient or workspace changes. Example: tearing 5 pieces of cheese and placing them = ONE action "tear off pieces of cheese and distribute them across the pizza", NOT 10 separate tear+place actions.
  GOOD EXAMPLES (right granularity):
  - "pick up the fork from the frying pan" (NOT reach → grasp → lift → withdraw as 4 actions)
  - "place the bowl on the refrigerator shelf" (NOT carry → lower → place → release as 4 actions)
  - "dice the tomato with repeated cross-cuts" (NOT 5 separate "cut left/center/right" actions)
  - "tear off pieces of cheese and scatter them across the pizza" (NOT 10 tear+place cycles)
  - "open the dishwasher door" (NOT grasp handle → pull → swing as 3 actions)
  - "stir the vegetables in the wok" (entire sustained stirring as one action)
  BAD EXAMPLES (too granular — will be REJECTED):
  - "reach toward the fork" + "grasp the fork handle" + "lift the fork" → should be ONE action
  - "lower the bowl" + "release the bowl onto the shelf" → should be ONE action
  - "press the knife into the tomato" + "draw the knife across" → should be ONE cut
  - "tear a piece of cheese" + "place it on pizza" repeated 5 times → should be ONE action
- MINIMUM SEMANTIC TEST: Before finalizing each atomic action, ask: "Does this action, by itself, accomplish something a human would recognize as a complete operation?" If NO, merge with its neighbor.
- You MUST produce at least 2 atomic actions per step.
- Typical: 3–8 atomic actions per step. Fewer is better when semantically justified.
- MAX SPAN: No single action should exceed 70%% of frames. Sustained same-patient operations (stirring, spreading, chopping) MAY span 30–60%%.
- WALKING/CARRYING: Treat "carry X from A to B" as ONE action unless trajectory spans >40%% of frames AND crosses a scene boundary (doorway, room change).
- MINIMUM SPAN: Every atomic action MUST span at least 2 frames.
- IDLE/STATIC BAN: Do NOT create atomic actions for periods where the actor's hands are stationary with no goal-directed physical operation in progress (resting, waiting, standing idle, observing). Absorb idle frames into the preceding or following goal-directed action. If the tail of the clip shows idle behavior after the step goal is achieved, extend the last goal-relevant action's end boundary to cover those frames.

CAUSAL CONTINUITY:
- The `patient` of atomic action i's end state must be consistent with the `patient`'s start state in atomic action i+1.
- If the actor switches hands or tools between actions, this MUST be a separate atomic action.
- The sequence of atomic actions must tell a coherent physical story: a human expert watching the clips should be able to reconstruct the full manipulation from the captions alone.
- PREVIOUS-STEP SPILLOVER: If the opening frames of this clip show the tail end of the PREVIOUS step's activity (a different goal than this step), do NOT create AAs for those frames. Begin the first AA at the point where THIS step's goal-directed activity visibly starts. Extend that first AA's start_frame_index backward to frame 1 for full coverage.
- STEP BOUNDARY RULE (CRITICAL): Every atomic action MUST belong to the goal of the CURRENT step. If a "NEXT step goal" is provided above, use it to detect boundary violations: any atomic action whose primary patient or action clearly belongs to the NEXT step's goal (not this step's goal) MUST be excluded. Concretely: if this step is about pouring milk and the next step is about taking a cloth from a drawer, then actions like "open the drawer" or "grasp the cloth" MUST NOT appear as atomic actions in this step — even if they are visible in the final frames of this clip. The last atomic action should complete or conclude the CURRENT step's goal. When in doubt, STOP the sequence at the last action that serves THIS step's goal.
- CAUSAL ORDER GATE: Read your planned AA sequence as a story. If an action requires a precondition that has not been established by earlier AAs in THIS step (e.g., tearing cheese before opening the cheese package, pouring from a sealed bottle), then the early frames showing that impossible action are spillover from another step — exclude those AAs and absorb their frames into the nearest valid AA.
- STEP TRANSITION RULE: If this is NOT step 1, the FIRST atomic action's `actor` and initial state description (in `caption`) MUST be consistent with the physical end-state implied by the previous step's last atomic action. If the previous step ended with the left hand holding a mug, this step must start with the left hand (not right hand) already holding the mug — unless the first AA explicitly describes a hand transfer.

POST-GENERATION SELF-CHECK (mandatory — run these checks on your draft output before emitting JSON):
1. KINEMATIC MERGE SCAN: If consecutive AAs form a kinematic chain targeting the SAME object with no visible pause between them (e.g., carry → lower → place → release; reach → grasp → lift), merge them into ONE action. Separate AAs are only justified by a VISIBLE PAUSE, a CHANGE OF INTENT, or a SWITCH TO A DIFFERENT OBJECT.
2. GOAL-RELEVANCE GATE: For each AA, ask: "Does this action directly serve THIS step's goal?" If NO (e.g., tidying an unrelated object, adjusting clothing, straightening a towel in a pouring step), remove it and absorb its frames into the nearest goal-relevant AA.
3. IDLE SCAN: If any AA describes static resting, waiting, or hands-off idle (no goal-directed motion), remove it and extend the neighboring action's boundary to cover those frames.
4. REPETITION SCAN: If two or more consecutive AAs share the same patient and describe the same operation (look for "continue", "repeat", "more", "additional", "resume", "further"), merge them into ONE.
5. CAUSAL ORDER SCAN: Read the AAs as a story. If an early AA requires a precondition not yet met (using an object before obtaining it, tearing material before unwrapping), those early AAs are spillover — remove and absorb.

GROUNDING REQUIREMENTS:
- All text fields MUST be grounded in visual evidence from the frames. Do NOT hallucinate objects, contacts, or states not visible.
- Do NOT describe hidden internal states (temperature, cleanliness, taste) unless visually confirmed (e.g., visible steam, visible dirt).
- Do NOT write future-step readiness statements in `caption` or `action` (e.g., "preparing for subsequent cooking", "ready for the next step").
- MATERIAL HALLUCINATION RULE: Do NOT assert specific material names (brass, chrome, stainless steel, oak, marble, copper, aluminum, ceramic, porcelain, etc.) unless the material is unambiguously identifiable from visual appearance alone. Use generic, visually-grounded descriptions instead (e.g., "metal handle" not "brass handle"; "dark wooden handle" not "oak handle"; "white bowl" not "porcelain bowl"). Color and shape are observable; exact material composition is not.
- Focus on observable physical primitives: positions, contacts, motions, orientations, and visible state changes.
- OBSERVABILITY RULE (for `caption` and `action` fields):
  Every physical property or state you assert MUST be directly observable (DO) in the frames:
  - DO: position, contact, orientation, open/closed state, color, shape, gross motion, container level, spatial arrangement — anything visible.
  - NDO: internal temperature, internal pressure, chemical composition, structural fatigue, exact weight, moisture content deep inside, flavor/taste.
  Do NOT assert NDO properties in `caption` or `action` unless visually confirmed (e.g., visible steam = hot). This is INTERNAL reasoning guidance — do not output DO/NDO labels in the JSON.
- Do NOT reference frame indices, timestamps, durations, or timecodes in `action`, `caption`, or `patient` fields. The only allowed frame reference is the integer `start_frame_index` / `end_frame_index` fields.
- Use consistent object naming across all atomic actions (match the step annotation's naming for `agent`, `patient`, and objects).
- CROSS-STEP NAMING INHERITANCE: The `patient` name for any object MUST match the name used in the reference step annotation's `causal_chain.patient` and `step_goal`. If the step annotation calls the object "striped cloth", every atomic action must also use "striped cloth" — not "cloth", "kitchen cloth", or "fabric". This also applies to objects mentioned in the step goal that serve as secondary patients across AAs.
- Avoid placeholders like "unknown", "N/A", "the object", "something".

ACTION-RELEVANCE FILTER (applies to ALL fields — action, caption, patient, actor):
- Every sentence in `action` and `caption` MUST be directly causally related to the physical operation being performed. Omit background objects, ambient scene details, and elements not involved in or affected by the atomic action.
- INCLUDE: objects directly manipulated, tools in use, surfaces providing direct support/contact, body parts executing the action.
- EXCLUDE: background furniture not involved, ambient lighting/weather, other people not participating, decorative items, general room layout, objects visible but not causally connected.
- SELF-CHECK: For every object or detail you mention, ask "Would removing this from the scene change whether the atomic action succeeds or fails?" If NO, omit it.

Output format (strict JSON only; no markdown, no comments, no extra text):
{{
  "step_id": <must match the step annotation step_id>,
  "atomic_actions": [
    {{
      "atomic_action_id": 1,
      "start_frame_index": 1,
      "end_frame_index": 9,
      "actor": "right hand",
      "action": "grasp the cup handle",
      "patient": "cup",
      "caption": "The right hand extends forward from the counter, approaches the ceramic cup handle from the right side, wraps fingers around it with thumb on top and three fingers underneath, establishing a stable pinch grip while the cup remains on the counter surface."
    }},
    {{
      "atomic_action_id": 2,
      "start_frame_index": 9,
      "end_frame_index": {num_frames + 1},
      "actor": "right hand",
      "action": "lift the cup off the counter",
      "patient": "cup",
      "caption": "The right hand, firmly gripping the cup handle, applies upward force to lift the cup approximately 10cm above the counter surface, breaking contact between the cup base and the counter."
    }}
  ]
}}

Additional constraints:
- `step_id` MUST match the step annotation's step_id exactly.
- `atomic_action_id` MUST start at 1 and increase by 1.
- All required keys MUST be present and non-empty (no empty strings, empty arrays, or null).
- Do NOT add any extra keys beyond the schema above.
- Output MUST be exactly one JSON object.

Now output the final strict JSON object only.
""".strip()


def build_stage2_boundary_verification_prompt(
    high_level_goal: str,
    draft_plan_outline: str,
    num_frames: int,
    current_boundaries_json: str,
) -> str:

    return f"""
You are an expert video step temporal boundary verifier.
You are given:
1) {num_frames} uniformly sampled frames from the FULL original video (chronological order), with frame labels.
2) A draft step plan (read-only).
3) A PROPOSED set of step boundaries that you must VERIFY and CORRECT if needed.

High-level goal: {high_level_goal}

Draft steps:
{draft_plan_outline}

Proposed boundaries (to verify/correct):
{current_boundaries_json}

Task:
For EACH boundary between consecutive steps, examine the frames AT and AROUND the boundary and determine whether the boundary is correctly placed.

For each step boundary (where step i ends and step i+1 begins), check:
- The last 2-3 frames of step i's clip: Do they show step i's action completing, or have they already started step i+1's action (reaching for a new object, new grip, new motion direction)?
- The first 2-3 frames of step i+1's clip: Do they show step i+1's action beginning, or is step i's action still ongoing (object still moving, hand still in contact)?

BLEEDING DETECTION (the primary error to catch):
- If the last frames of step i's clip show ANY of the following, the boundary is TOO LATE — move it EARLIER:
  (a) The agent's hand/arm is REACHING TOWARD step i+1's target object
  (b) The agent's body/head has TURNED or SHIFTED orientation toward step i+1's workspace
  (c) A NEW GRIP is being established on a DIFFERENT object than step i's patient
  (d) The agent is in a WALKING/LOCOMOTION phase moving toward a new workspace area
  (e) Step i's primary object has been RELEASED and the hand is already moving AWAY from it toward a new target
- The ONLY acceptable content in step i's last frames is: step i's action completing, the object in its final position, or a brief neutral/idle pause BEFORE any new motion begins.
- If the first frames of step i+1's clip show the agent still COMPLETING the action of step i (object still in motion, hand still gripping previous object), the boundary is TOO EARLY. Move it LATER — but ONLY if this does not cause any of the bleeding patterns (a)-(e) above.

BOUNDARY CORRECTNESS CRITERIA:
- CORRECT boundary: The last frame of step i shows step i's action completed (object in final position, hand released or withdrawing, or neutral pose). The first frame of step i+1 shows the beginning of a new action or transition toward it.
- INCORRECT (boundary too late): The last frame of step i shows the agent already reaching toward step i+1's target object. FIX: move boundary earlier.
- INCORRECT (boundary too early): Step i's action is visibly incomplete in its clip (object not yet in final position). FIX: move boundary later, but ONLY if this does not cause next-step bleeding.

ASYMMETRIC CORRECTION RULE (NON-NEGOTIABLE):
- When in doubt, move boundaries EARLIER (trim the end of the current step) rather than LATER.
- It is acceptable to lose 1-2 tail frames of a step, but UNACCEPTABLE to include any next-step action frames.
- If a boundary is ambiguous and you cannot clearly determine the correct position, KEEP IT UNCHANGED.

CONSTRAINTS (same as original localization — do not violate):
- start_frame_index of the first step MUST be 1.
- end_frame_index of the last step MUST be {num_frames + 1}.
- Contiguous: end_i == start_{{i+1}} for consecutive steps. No gaps, no overlaps.
- start_frame_index < end_frame_index for each step.
- All indices are integers in [1, {num_frames + 1}]; {num_frames + 1} only allowed for last end_frame_index.
- Output must contain exactly one entry per step_id.

Output format (strict JSON only; no markdown, no extra text):
{{
  "steps": [
    {{
      "step_id": 1,
      "start_frame_index": 1,
      "end_frame_index": <corrected_or_unchanged>
    }},
    {{
      "step_id": 2,
      "start_frame_index": <must_equal_previous_end>,
      "end_frame_index": <corrected_or_unchanged>
    }}
  ]
}}
""".strip()


def build_stage2_boundary_refinement_prompt(
    step_i_goal: str,
    step_i_plus_1_goal: str,
    num_dense_frames: int,
    current_boundary_description: str = "",
) -> str:

    return f"""
You are an expert video step boundary specialist.
You are given {num_dense_frames} densely sampled frames from a narrow time window around a SUSPECTED STEP BOUNDARY (chronological order, with frame labels).

The two steps meeting at this boundary:
- ENDING step (step i): {step_i_goal}
- STARTING step (step i+1): {step_i_plus_1_goal}

{f"Current boundary estimate: {current_boundary_description}" if current_boundary_description else ""}

Task:
Examine these dense frames and identify the EXACT transition point — the frame where step i's action has COMPLETED and step i+1's action has NOT YET BEGUN.

WHAT TO LOOK FOR:
- The COMPLETION of step i: the moment when the primary object reaches its final resting position, the hand has released or is withdrawing, or the described world-state change is fully visible and stable.
- The INITIATION of step i+1: the first visible motion toward step i+1's target object (reaching, turning, shifting body orientation).
- The boundary frame should be the LAST frame that still belongs to step i (showing completion), NOT the first frame of step i+1.
- ANTI-LEAKAGE CHECK: In the boundary frame you choose, the agent's hands/body MUST NOT show ANY motion toward step i+1's target object. If you see even the beginning of a reach, turn, or weight shift toward the next action, move the boundary EARLIER.
- WHAT "COMPLETION" LOOKS LIKE: The primary object is in its final resting position for this step. The hand has released contact OR is stationary on the object (no forward motion). The body posture is neutral or still oriented toward step i's workspace. There is NO visible anticipatory motion toward the next task.

DECISION RULE:
- If you can clearly identify the transition, report the frame where step i is complete.
- If the transition is ambiguous, prefer the EARLIER frame (it is better to end step i slightly early than to include step i+1's actions in step i's clip).
- If no clear boundary is visible in these frames (both steps seem to blend), report the middle frame.

Output (strict JSON only; no markdown, no extra text):
{{
  "refined_boundary_frame_index": <1-based index into the {num_dense_frames} provided frames>,
  "confidence": "<high|medium|low>"
}}
""".strip()
