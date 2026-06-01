FINAL_CAUSAL_TRACE_PROMPT_ID = "reviewer_grade_task_specific_causal_trace_prompts"

FINAL_SHARED_SYSTEM_PROMPT = """
You are generating reviewer-grade causal reasoning traces for embodied multimodal QA.

The trace is not a decorative explanation and not a paraphrase of the answer. It is the
causal audit trail that should convince a strict benchmark reviewer that the answer
follows from the evidence through task-specific physical, temporal, and procedural
logic.

Universal quality contract:
- Ground every claim in the provided visual evidence, structured fields, and plan
  context. Do not introduce objects, steps, outcomes, intentions, or hidden events that
  are not supported by those inputs.
- Build explicit causal links in the form: observed condition -> physical or procedural
  mechanism -> state change -> enabled, blocked, or required next condition.
- Name the load-bearing object relations: agent, tool, patient, container, support
  surface, opening, handle, edge, contact patch, trajectory, alignment, containment,
  clearance, obstruction, material property, and resulting state whenever relevant.
- Separate observation from inference. First identify the visible or structured fact,
  then state why it causally matters.
- Include a concrete counterfactual, removal test, skip test, or alternative-action
  disambiguation. The test must say what would fail physically or procedurally, not only
  that the answer would change.
- Respect the task type. A goal-recognition trace should not read like a precondition
  checklist; a feasibility trace should not read like a story summary; a recovery trace
  should not read like ordinary next-step prediction.
- Keep the final answer outside the trace. Do not copy the answer sentence, do not end
  with "therefore the answer is", and do not use the trace as a verbose restatement.
- Write natural analytical prose in paragraphs. Do not use bullets, numbered lists,
  headings, XML tags, markdown, path names, frame names, JSON names, or meta-commentary.
- If the evidence is sparse, explicitly reason from the available state and name the
  uncertainty boundary; do not fill missing visual details with inventions.

Output only the causal reasoning trace text. The data pipeline will wrap it in
<think>...</think> and append the target answer separately.
"""

FINAL_USER_PROMPT_TEMPLATE = """
=== REVIEWER-GRADE CAUSAL TRACE TASK ===
Task name:
{task_name}

Benchmark dimension:
{dimension}

Task-specific final prompt:
{task_specific_prompt}

=== QUESTION ===
{question}

=== STRUCTURED VISUAL AND CAUSAL EVIDENCE ===
{llm_fields}

=== PLAN CONTEXT ===
{plan_context}

=== TARGET ANSWER KEPT OUTSIDE THE TRACE ===
{answer}

Write only the causal reasoning trace. Start from concrete evidence, expose the
task-specific causal mechanism, use the required disambiguation or failure test, and
stop after the decisive causal logic. Do not output tags. Do not copy the target answer.
"""

FINAL_DIMENSION_PROMPTS = {
    "Composition": """
Composition tasks test whether the model can assemble perceptual facts, local step
semantics, temporal order, and plan context into a coherent procedural interpretation.
The trace must expose how smaller observations combine into a larger goal, action,
dependency, or predicted sequence. It should name the intermediate state that each
sub-action creates and show why that state is the bridge to the next sub-action.
Avoid flat summaries. The reviewer should see a compositional chain: parts -> roles ->
state transitions -> ordered plan logic -> rejected alternative composition.
""",
    "Executability": """
Executability tasks test whether the action can physically happen under the current
spatial and affordance conditions. The trace must behave like a feasibility proof. It
must separately identify spatial preconditions, functional object affordances, contact
or force pathways, and the exact sub-motion that would fail if a condition were absent.
Do not merely say objects are "ready" or "in the right place"; specify reachability,
alignment, clearance, support, containment, opening, graspability, rigidity, sharpness,
flexibility, stability, or unobstructed access as needed.
""",
    "Effects": """
Effects tasks test whether the model understands what an action changes. The trace must
make the before-state explicit, identify the physical mechanism that drives the change,
and name the after-state in spatial and/or affordance terms. It should connect the
postcondition to later plan usefulness and include a failure contrast showing what
would remain unchanged if the mechanism did not occur.
""",
    "Robustness": """
Robustness tasks test whether the model can reason under disrupted, missing, flawed, or
counterfactual plan conditions. The trace must first reconstruct the normal causal
chain, then locate the exact broken link, propagate immediate and downstream effects,
and either diagnose the failure or justify the recovery. It should not simply state
that something goes wrong; it must say which precondition is lost, which mechanism no
longer runs, and which later state cannot be reached.
""",
}

FINAL_TASK_PROMPTS = {
    "Task_08_Goal_Recognition": """
Dimension: Composition.
Trace purpose: infer the high-level goal from the full temporal composition of the
video.

The trace must:
- Treat the video as an ordered chain of subgoals, not a bag of actions.
- Identify the recurring manipulated objects, tools, containers, surfaces, or target
  areas and explain their roles across the sequence.
- For each major phase, state the state change it creates and how that state makes the
  next phase possible.
- Explain the single higher-level objective that makes the subgoals mutually coherent.
- Reject at least one weaker interpretation, such as random handling, mere inspection,
  cleanup, or isolated manipulation, by naming the causal evidence that rules it out.

Reviewer bar: the trace should make the overall goal feel inevitable from the ordered
state transitions, not inferred from one salient object or copied from the answer.
""",
    "Task_09_Macro_Anchor_Extraction": """
Dimension: Composition.
Trace purpose: identify the causally central anchor objects in the episode.

The trace must:
- Evaluate each candidate object by causal role: tool, patient, container, support,
  target, source, destination, fastener, boundary, or bystander.
- Use a removal test for every selected anchor: if this object were absent, blocked, or
  replaced by an inert bystander, which step would fail and why.
- Distinguish direct manipulation from background salience. Large, colorful, or nearby
  objects count only if they carry a causal dependency.
- Explain multi-object coupling when anchors work together, such as tool-on-patient,
  container-receives-object, support-stabilizes-workpiece, or source-to-destination
  transfer.
- Avoid ending as a copied list. The final paragraph should synthesize why the selected
  anchors are the load-bearing set.

Reviewer bar: the trace should show object centrality through necessity and role, not
through visual prominence.
""",
    "Task_10_Clip_to_Step_Goal": """
Dimension: Composition.
Trace purpose: infer the local step goal from a clip by connecting visible motion to
the intended state change.

The trace must:
- Start with the contact path: which hand/tool contacts which object, from where to
  where, with what trajectory or containment relation.
- Identify the acted-on object and the patient state before the motion.
- Explain the state change produced by the motion: transfer, alignment, separation,
  covering, opening, insertion, removal, mixing, fastening, cleaning, or repositioning.
- Connect that local state change to the broader plan: what next precondition it
  creates or preserves.
- Rule out a plausible alternate step goal by naming the missing motion, missing
  object, or wrong postcondition for that alternative.

Reviewer bar: the trace should infer the goal from mechanism and postcondition, not
from surface-level action words.
""",
    "Task_11_Action_Phrase": """
Dimension: Composition.
Trace purpose: derive the precise action phrase from kinematics and object response.

The trace must:
- Analyze trajectory, grip, speed, force direction, contact interface, and duration.
- Compare nearby verbs. For example, separate push from press, place from drop, stir
  from scrape, align from attach, lift from remove, open from pull aside, or inspect
  from manipulate.
- Explain how the patient object responds: displacement, rotation, deformation,
  containment, exposure, fastening, cutting, mixing, or stabilization.
- State why a different verb would require a different contact pattern or resulting
  state.
- Preserve the exact action granularity: not too broad as a whole task, not too narrow
  as a hand twitch.

Reviewer bar: the trace should justify the verb phrase through observable mechanics.
""",
    "Task_04_Affordance_Visual_Semantics": """
Dimension: Effects.
Trace purpose: ground a local affordance or hotspot in physical properties and
interaction mechanism.

The trace must:
- Name the object part or hotspot and the relevant geometry, material, opening, edge,
  handle, surface, cavity, flexible region, rigid region, sharp region, or graspable
  region.
- Explain how that property supports a specific contact type: grasping, pressing,
  pulling, cutting, pouring, inserting, scraping, supporting, containing, or aligning.
- Trace force or motion through the affordance to the resulting interaction.
- Include an altered-property test: what would fail if the object were closed, too
  smooth, too soft, too rigid, full, empty, misaligned, dull, unstable, or inaccessible.
- Connect the affordance to the task effect, not only to a label.

Reviewer bar: the trace should make the affordance a causal mechanism, not a visual
tag.
""",
    "Task_12_State_Evolution": """
Dimension: Composition.
Trace purpose: explain a state transition over time.

The trace must:
- Inventory the before-state: position, orientation, containment, configuration,
  surface contact, openness, mixture state, cleanliness, attachment, or amount.
- Identify the force source and contact path that drive the transition.
- Name the physical principle when relevant: gravity, friction, leverage, cutting,
  compression, containment, support, fluid flow, adhesion, heat, mixing, or alignment.
- Describe the after-state as an actual changed condition, not only as completion.
- Explain what the after-state enables, blocks, exposes, secures, loosens, cleans, or
  prepares for the next step.
- Include a no-change contrast: if the key action did not occur, which before-state
  would persist.

Reviewer bar: the trace should be a mechanism of change, not a chronological caption.
""",
    "Task_05_Holistic_Causal_Chain": """
Dimension: Effects.
Trace purpose: reconstruct the full causal chain from preconditions through action to
postconditions.

The trace must:
- Separate spatial preconditions from affordance preconditions before describing the
  action.
- Name agent, action, tool if any, patient, source, destination, and support or
  containment structure.
- Explain the mechanism that links them: contact, force transfer, trajectory,
  alignment, containment, resistance, deformation, rotation, or support.
- Distinguish spatial effects from affordance effects after the action.
- Show how the postconditions become useful for the next procedural state.
- Run a skip, removal, or wrong-alignment test to prove the chain is necessary.

Reviewer bar: the trace should read like a complete causal graph in prose.
""",
    "Task_13_Strategic_Rationale": """
Dimension: Composition.
Trace purpose: prove why a step is strategically necessary in the plan.

The trace must:
- Identify the postconditions created by the step.
- Name the later step, final objective, or plan constraint that depends on those
  postconditions.
- Explain why the dependency is causal rather than merely temporal.
- Run a skip test: if this step were omitted, what is the first downstream failure and
  which object/state/precondition is missing.
- Explain whether the step is unique, earliest, safest, or most direct as a provider of
  the needed condition.
- Avoid generic phrases like "it prepares for the next step" unless the prepared state
  is explicitly named.

Reviewer bar: the trace should justify necessity through downstream preconditions.
""",
    "Task_01_Spatial_Precondition": """
Dimension: Executability.
Trace purpose: state the spatial arrangements required before the action can execute.

The trace must:
- Name exact positions, orientations, distances, alignments, openings, clearances,
  support contacts, containment relations, and source-destination layout.
- Tie each relation to reach, line of motion, collision avoidance, stable support,
  visibility, insertion path, transfer path, or containment.
- Explain which sub-motion would fail if each relation were absent.
- Distinguish spatial readiness from affordance readiness. For example, "near the
  hand" is spatial; "graspable handle" is affordance.
- Include at least one wrong-layout contrast, such as too far, rotated away, blocked,
  unsupported, off-center, closed path, or insufficient clearance.

Reviewer bar: the trace should make space a set of executable constraints, not a
generic scene description.
""",
    "Task_02_Affordance_Precondition": """
Dimension: Executability.
Trace purpose: state the functional object states required before the action can
execute.

The trace must:
- Name each required affordance separately: open, closed, empty, filled, graspable,
  rigid, flexible, sharp, dull, stable, loose, attached, separable, aligned, exposed,
  reachable, clean, hot, powered, unlocked, or unobstructed as appropriate.
- For every property, trace the dependent sub-motion: where force is applied, what
  surface or part carries contact, and what response the property permits.
- State the provenance of important properties when possible: already true, created by
  a previous step, maintained by the current setup, or about to be consumed.
- Run at least two concrete failure tests for missing properties, describing the failed
  physical motion rather than only the missing label.
- End by tying the properties into joint affordance readiness.

Reviewer bar: the trace should be a functional feasibility proof with mechanisms.
""",
    "Task_03_Physical_Feasibility": """
Dimension: Executability.
Trace purpose: verify whether the step is physically possible now.

The trace must:
- Check spatial preconditions and affordance preconditions separately before combining
  them.
- Explain how the spatial layout creates a viable reach, transfer, insertion, support,
  containment, or tool-use path.
- Explain how the object properties allow the needed force, grip, deformation,
  cutting, pouring, fastening, opening, or stabilization.
- Combine the checks into a feasibility verdict grounded in this moment, not in the
  general task.
- Include a single-condition failure contrast: if one checked condition were false,
  which sub-action would fail first.

Reviewer bar: the trace should make feasibility the conclusion of explicit checks.
""",
    "Task_06_Spatial_Postcondition": """
Dimension: Effects.
Trace purpose: explain the spatial relationship created by the action.

The trace must:
- Recall the relevant initial layout without over-summarizing the whole video.
- Trace the displacement, rotation, transfer, insertion, removal, alignment, stacking,
  containment, exposure, support change, or clearance creation.
- Name the new spatial relation precisely: inside, on top of, beside, aligned with,
  attached to, separated from, cleared from, closer to, oriented toward, or accessible
  from.
- Explain what this new layout enables or prevents in the next plan step.
- Include a contrast: if the spatial postcondition did not occur, what later motion or
  contact path would be blocked.

Reviewer bar: the trace should show a spatial before/after transformation.
""",
    "Task_07_Affordance_Postcondition": """
Dimension: Effects.
Trace purpose: explain the functional capability or limitation created by the action.

The trace must:
- Recall the pre-action affordance state.
- Identify the physical cause of the affordance change: opening, fastening, cutting,
  filling, emptying, cleaning, heating, mixing, loosening, tightening, exposing,
  wetting, drying, or stabilizing.
- Name the new capability or limitation created after the action.
- Explain how that new affordance state becomes a precondition for a later step.
- Include a counterfactual: if the action stopped short, which capability would still
  be unavailable.

Reviewer bar: the trace should explain a functional transformation, not merely a
visible placement change.
""",
    "Task_14_Inter_Step_Dependency": """
Dimension: Composition.
Trace purpose: show how one step's effects satisfy another step's preconditions.

The trace must:
- Identify the earlier step's spatial effects and affordance effects separately.
- Identify the later step's spatial and affordance preconditions separately.
- Match each earlier effect to the later precondition it satisfies.
- Explain why the dependency is directional: the later step needs the earlier state,
  not merely the same objects.
- Run a removal test: if the earlier step were absent, which exact precondition would
  fail and what later sub-motion would become impossible.

Reviewer bar: the trace should make inter-step dependency a state-matching proof.
""",
    "Task_15_Next_Step_Prediction": """
Dimension: Composition.
Trace purpose: predict the next step from the cumulative prefix state.

The trace must:
- Inventory the boundary state after the prefix: completed objects, unfinished objects,
  current locations, active tools, available containers, exposed surfaces, and
  functional states.
- Identify the next unmet goal condition in the plan.
- Explain why the current state now satisfies the preconditions for the predicted next
  step.
- Compare at least two candidate next actions. Reject the weaker alternative by naming
  the missing precondition, premature ordering, redundant postcondition, or downstream
  mismatch.
- Explain why the chosen step is the earliest causally valid successor.
- Connect the chosen step's expected postcondition to a later plan requirement.

Reviewer bar: the trace should be next-step inference from state, not a guess from
frequency or script memory.
""",
    "Task_16_Middle_Steps_Infill": """
Dimension: Composition.
Trace purpose: infer missing middle steps by bridging head postconditions to tail
preconditions.

The trace must:
- State what the head segment leaves true.
- State what the tail segment requires before it can begin.
- Identify every state gap between those two observations: missing object, missing
  placement, missing opening, missing mixture, missing cleaned state, missing assembled
  part, missing tool readiness, or missing containment.
- Justify each inferred middle step as creating one required bridge condition.
- Preserve order by matching each inferred postcondition to the next inferred
  precondition.
- Reject an over-short or reordered bridge if it would leave a tail precondition
  unsatisfied.

Reviewer bar: the trace should make the invisible middle necessary from boundary
states.
""",
    "Task_17_Next_K_Steps_Prediction": """
Dimension: Composition.
Trace purpose: forecast multiple next steps as a chained sequence.

The trace must:
- Describe the state after the prefix with enough detail to constrain the first
  predicted step.
- For each predicted step, name the condition that makes it possible now and the
  postcondition it creates for the following step.
- Keep the sequence causally tight: no step should appear before its prerequisites or
  after its postcondition is already unnecessary.
- Explain how the full predicted chain advances the overall goal.
- Reject at least one tempting sequence error, such as skipping a setup step, repeating
  a completed action, or jumping to a downstream action too early.

Reviewer bar: the trace should show a linked plan rollout, not a list of likely actions.
""",
    "Task_18_Bad_Plan_Diagnosis_And_Repair": """
Dimension: Robustness.
Trace purpose: diagnose a flawed plan and explain how the repair restores causal
continuity.

The trace must:
- Walk through the plan in order, checking preconditions and postconditions at each
  link.
- Locate the first broken link, not just the most visually obvious odd step.
- Name the flaw type: impossible precondition, wrong object, wrong order, missing
  setup, missing cleanup, redundant action, impossible transition, or unsupported
  postcondition.
- Explain the mechanism of failure: which state is not created, which contact path
  cannot run, or which later precondition remains false.
- Explain the repair as a causal restoration, matching each inserted, removed, or
  reordered step to the condition it fixes.
- Verify that the repaired sequence now reaches the later plan state.

Reviewer bar: the trace should look like plan debugging over a causal state graph.
""",
    "Task_19_Counterfactual_Outcome": """
Dimension: Robustness.
Trace purpose: predict the outcome under a counterfactual disruption.

The trace must:
- First reconstruct the normal chain: precondition, action mechanism, immediate
  postcondition, and downstream use.
- Identify the exact counterfactual break point and classify it as spatial, affordance,
  force/contact, material, temporal, or procedural.
- Propagate consequences through immediate effect, secondary effect, and final
  plan-level outcome.
- Contrast the normal end-state with the degraded counterfactual end-state.
- Avoid vague statements like "the task would fail"; specify what remains unmade,
  unmoved, uncleaned, unfastened, inaccessible, unsupported, or unusable.

Reviewer bar: the trace should be a counterfactual propagation proof.
""",
    "Task_20_Failure_Recovery": """
Dimension: Robustness.
Trace purpose: explain how to recover from a failure state and resume the plan.

The trace must:
- Describe the failure configuration precisely: wrong object state, misalignment,
  spill, obstruction, unstable support, lost grip, missing tool, blocked opening,
  failed attachment, or wrong location.
- Diagnose the physical or procedural cause of the failure.
- Map each recovery action to the precondition it restores.
- Explain why the recovery order matters.
- Verify that after recovery, the original action or next plan step can proceed through
  the intended mechanism.
- Reject a superficial fix if it leaves the broken precondition unresolved.

Reviewer bar: the trace should make recovery a precondition-restoration protocol, not
generic advice.
""",
}

FINAL_RETRY_PROMPT_TEMPLATE = """
The previous trace did not meet the reviewer-grade causal contract.

Failure reason:
{failure_reason}

Rewrite the trace from scratch for task:
{task_name}

Use the task-specific final prompt again:
{task_specific_prompt}

Mandatory repair:
- Add the missing causal mechanism, not a longer paraphrase.
- Ground each inference in evidence or plan context.
- Include the required counterfactual, removal, skip, or alternative-action test.
- Keep the target answer outside the trace and do not copy it.
- Output plain analytical prose only.
"""
