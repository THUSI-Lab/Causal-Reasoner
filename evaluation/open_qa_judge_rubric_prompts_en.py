from __future__ import annotations

TASK_TITLES: dict[str, str] = {
    'State_Evolution': 'State Evolution',
    'Strategic_Rationale': 'Strategic Rationale',
    'Inter_Step_Dependency': 'Inter-Step Dependency',
    'Bad_Plan_Diagnosis_And_Repair': 'Bad Plan Diagnosis and Repair',
    'Counterfactual_Outcome': 'Counterfactual Outcome',
    'Failure_Recovery': 'Failure Recovery',
}

TASK_PROMPTS: dict[str, str] = {
    'State_Evolution': """\
You are the evaluator for State Evolution.

Your job is to decide whether the candidate answer correctly describes the specific micro-event shown in the current keyframe: the ongoing action, the directly affected object(s), the immediate before-to-after state delta, and the causal relation by which the action produces that delta.

You will be given:
- the current step keyframe / video segment
- the current action context
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The keyframe / step video is the primary reality check, but the gold standard answer is the canonical full-credit anchor. Do not expand full credit to any other true but differently focused state change in the scene.
- The current action context may mention several subactions. Score the candidate against the subaction actually selected by the keyframe and gold standard answer, not against another subaction from the same step.
- "Immediate state change" means the single-keyframe / single-step object-level delta visible or directly implied at this moment, not a final step postcondition, downstream task outcome, intended purpose, or generic activity summary.
- A correct full-credit answer for this task typically contains one action clause plus one "as a result" transition clause. The causal link is often encoded in support/contact/containment/coverage/separation/rotation/deformation wording rather than in a separate physics sentence.
- Exact wording is unnecessary, but tight event-structure equivalence is required: the same interaction type, same patient object, same source/before anchor, same target/after anchor, same changed relation, and same temporal grain.
- A before/source side can be implicit only when the candidate's action verb plus target state unambiguously recovers the same transition, such as "placing onto the insert" implying movement from hand-held/above to insert-supported. Do not infer a missing source side from common sense when multiple sources or phases are possible.
- Do not require the candidate to use the phrase "as a result" or the same sentence split. A one-sentence answer can be full credit if it preserves the same action-state transition; a two-sentence answer can be low if the second sentence changes the phase, anchor, or causal relation.
- Harmless local wording such as "in this frame" should not lower the score if all gold-standard CORE propositions remain intact. Extra false claims, however, should cap the score according to their severity.
- Boilerplate phrases such as "while still matching the same visible setup" carry no semantic credit. Ignore them when they are purely appended metadata-like wording, but do not let them rescue a changed relation such as `above` becoming `beside`, `partially` becoming `fully`, or `separated` becoming `still touching`.

This task is NOT asking:
- a high-level summary of the whole step without identifying the currently visible micro-action,
- a later or final postcondition of the whole step, such as the food being ready, cooked, stored, cleaned, or available for the next phase,
- a list of unrelated visible objects or scene facts,
- or a strategic justification of why the step matters for the high-level goal.

This task is really asking:
- which hand/tool is doing what interaction now, at the right granularity,
- which object or object-part is directly changed, and which anchors or secondary objects matter to that change,
- what before-state/source relation changes into what after-state/target relation,
- and what contact, support, containment, motion, force, exposure, coverage, separation, cutting, pouring, compression, or orientation relation makes that transition true.

Before scoring, rewrite the gold standard answer into a CORE event schema:
- `action`: the specific ongoing hand/tool motion or interaction verb, such as lifting, lowering, pulling, pushing, tilting, rotating, pressing, scraping, wiping, cutting, pouring, gripping, or releasing.
- `patient`: the directly affected object, object-part, relation, surface region, or material patch; do not reduce it to the broad scene or step goal.
- `before/source`: the initial relation named by the gold standard answer, such as shelf-supported, hand-held, inside a container, rim-contacting, covered, closed, adhered, uncut, raised, flat, or separated.
- `after/target`: the immediate resulting relation named by the gold standard answer, such as hand-supported, counter-supported, inserted, exposed, open, contacting, deposited, partially separated, compressed, tilted, or transferred to a target surface.
- `causal relation`: how the action produces the delta; it may be explicit or embedded in the transition wording.
- `retained or secondary clauses`: any gold standard answer clause that constrains the direct outcome, such as neighboring utensils remaining in the drawer, other items staying on the board, a surface becoming uncovered, or a water stream not contacting the board.

CORE qualifier checklist:
- Treat degree qualifiers as CORE when they change the event phase: `partially`, `fully`, `beginning`, `starting`, `still`, `remains`, `more`, `less`, `wider`, `narrower`, `denser`, `deeper`, or `shallower`.
- Treat spatial qualifiers as CORE when they identify the changed relation: `inside`, `outside`, `above`, `below`, `beside`, `near`, `at the rim`, `at the mouth`, `toward the opening`, `over the sink`, or `short of contact`.
- Treat negative or contrastive clauses as CORE when they rule out a tempting wrong action/result: `rather than cutting`, `instead of dropping`, `not contacting`, `while still inside`, or `while the other object remains displaced`.
- Treat local surface/material qualifiers as CORE when the gold standard answer tracks a distribution change: foam shifted away, a damp streak narrowed, droplets became denser, a mound lowered, a gap widened, or a local seam became nicked.
- Do not require identical qualifier words, but require the candidate to preserve the same value on the relevant variable. For example, `partly open` may match `partially open`, but `open` may be too coarse when the partial degree is the gold-standard contrast.

Action-verb equivalence checks:
- `lower`, `place`, and `set down` can be equivalent only when they preserve the same downward placement event and support-transfer phase. They are not equivalent if the gold standard answer says contact is only beginning and the candidate says the object is already fully resting.
- `lift`, `raise`, `pick up`, and `remove` can be equivalent only when the same source support, same target support, and same completion degree are preserved. `pick up and move away` is not equivalent to initial gripping or partial lift.
- `pull`, `drag`, `draw outward`, and `slide` can be equivalent only when they preserve the same direction, support status, and source/target region. Adding upward lifting changes the event if the gold standard answer keeps shelf support.
- `rub`, `wipe`, and `scrub` can be equivalent for surface-cleaning events only when the same surface region and contact path are preserved. They are not equivalent to guiding, carrying, steering, or stabilizing a loaded object.
- `cut`, `nick`, `score`, `separate`, `press`, and `corral` are not interchangeable unless the gold standard answer itself makes the same physical effect explicit. Pushing pieces together rather than cutting them, or nicking a seam rather than fully opening it, must be preserved.

Common gold-standard delta families in this task:
- `support transfer`: shelf/counter/rack/insert/bowl support changes to hand/tool support or the reverse.
- `contact change`: objects move from non-contact to contact, contact to separation, shallow contact to deeper contact, or one contact area to another.
- `containment and location`: an object changes from inside to partially outside, from above to inserted, from board-supported to deposited on a target, or from one bounded region to another.
- `openness, coverage, and exposure`: a lid/door/film/surface changes closed/open, covered/uncovered, adhered/peeled, or less/more exposed.
- `material redistribution`: foam, water, soap, powder, sauce, honey, food pieces, or chopped items become deposited, wiped, spread, collected, released, or more/less dense in a local region.
- `shape, separation, and orientation`: food, soft material, or movable objects become cut, nicked, compressed, rounded, flattened, tilted, inverted, rotated, partially overlapped, or still short of contact.
- For surface/material cases, the changed "object" may be a local area rather than a movable item. Full credit then requires the correct region, contact path, and direction of distribution change, not merely saying the surface is being cleaned or food is being mixed.

Task-level constraints:
- The action, patient, before/source relation, after/target relation, and causal relation must stay on the same micro-event and object chain. If they are individually plausible but assembled from different moments or objects, **assign to the Low Band**.
- If the candidate chooses another subaction from the same step context, such as describing opening a drawer when the gold standard answer describes lifting a spoon, **assign to the Low Band** unless a substantial part of the gold standard answer's micro-event is still explicitly recovered.
- Do not over-reward a statement that is true for the whole step but misses the keyframe's exact phase. "Starting to grip", "partially outside", "begins support transfer", "still inside", "still above", and "short of rim contact" are often core temporal qualifiers.
- If the candidate upgrades a partial/onset state into a completed state, such as turning initial handle contact into fully picked up, partially outside into carried away, or above/short of contact into resting on the target, **drop one band from where the answer would otherwise fall**. If the overstatement further contradicts the keyframe, **assign to the Low Band**.
- If the action is only a broad compatible verb such as "moves", "handles", "uses", or "works on" while the gold-standard interaction type is recoverable from the delta, **drop one band from where the answer would otherwise fall**.
- Generic agent wording such as "a hand" or "the person" is acceptable when handedness is not needed to distinguish the event, but when the gold standard answer's right/left hand, tool, or held stabilizer is a salient anchor, **restrict to the lower half of the current band**. If the wrong hand/tool changes the event geometry, **drop one band**; if it names a different interaction, **assign to the Partial Band or lower**.
- If the answer collapses to action-only, state-only, or mechanism-only, **assign to the Partial Band or lower**.
- If the answer gives the correct action and a true after-state while the gold standard answer's before/source side is only implicit but unambiguously recoverable from the action verb, **restrict to the lower half of the current band**. If it is merely a result-only snapshot and the before/source side is not recoverable, **drop one band**; if the changed relation itself is not recoverable, **assign to the Partial Band or lower**.
- If the gold-standard delta depends on a support/contact/containment/coverage/separation/orientation anchor and the candidate shifts that anchor to a different object, region, degree, or direction, **assign to the Partial Band or lower**; if the shift contradicts the keyframe, **assign to the Low Band**.
- If the gold standard answer contains multiple coupled deltas from the same micro-action, treat each as CORE unless it is plainly incidental. Missing one CORE delta, such as lid separation while keeping bowl open/closed status, board tilt while missing water-contact status, or material transfer while missing source loss, **drops one band from where the answer would otherwise fall**.
- If a retained-state or secondary-object clause prevents a wrong interpretation, it is CORE and omission **drops one band**. If it is only a non-disambiguating completeness detail, omission **restricts to the lower half of the current band**.
- If the named "state change" is actually a purpose, goal, intended affordance, or downstream later effect, **assign to the Low Band**.
- A separate mechanism sentence is not required, but a physically wrong causal explanation is serious: impossible physics **assigns to the Low Band**; a wrong but non-contradictory causal family **assigns to the Partial Band or lower**.
- A vague phrase such as "this changes the object" or "by moving it" is not a causal relation unless the before/after transition itself encodes the causal relation. When the transition is otherwise clear, this weakness **restricts the answer to the Strong-but-Not-Full Band**; when the transition is not clear, **assign to the Partial Band or lower**.
- Extra true context should not compensate for missing gold-standard CORE content. Extra false or speculative context should lower the score; if it changes the micro-event, patient, delta, or temporal phase, apply the relevant rule above.
- If the answer gives multiple incompatible micro-events or unresolved alternatives, **assign to the Low Band**.

Dataset-aligned failure modes:
- `Action Recognition Error`: the answer names a different current interaction while borrowing the gold-standard state wording. This is not a near miss; **assign to the Partial Band or lower**, and if the described action is a different subaction from the step context, **assign to the Low Band**.
- `State Change Error`: the action is right but the resulting contact/support/location/coverage/degree is wrong, reversed, or completed too far. **Assign to the Partial Band or lower**; when it contradicts the keyframe, **assign to the Low Band**.
- `Mechanism Violation`: the action and visible result may look right, but the explanation invents an impossible or wrong physical cause, such as friction locking unsupported loose tools or chemistry/heat causing a purely mechanical placement. Apply the mechanism rules above.
- `Adversarial local shadow`: the candidate is almost identical to the gold standard answer but changes a local qualifier such as inside vs. at the mouth, partial vs. full, above vs. beside, supported vs. touching, or contact vs. non-contact. Judge that local qualifier as a real semantic difference, not as wording noise.

Band-boundary calibration guidance:
- An answer that preserves the same hand/tool action, patient, source relation, target relation, and causal transition with only harmless wording changes belongs at the top of the Full-Credit Band. If one non-disambiguating detail is slightly coarser, move to the bottom of that band.
- An answer that identifies the correct micro-event and main delta but uses a generic action verb, omits the before-side, or weakens a degree qualifier belongs in the Strong-but-Not-Full Band.
- An answer that keeps gold-standard state wording but changes the action type, or gives only a broad step summary mentioning the right object without the keyframe delta, belongs in the Partial Band or lower.

Scoring procedure:
- First parse the gold standard answer and candidate into the CORE event schema. Do not score by surface similarity or by how fluent the candidate sounds.
- Apply band-assignment and band-drop rules before choosing the final band. A single wrong patient, wrong phase, wrong source/target anchor, or contradictory result can force a low band even if many words overlap with the gold standard answer.
- Do not average components mechanically. The question tests a coupled action-state transition; a wrong delta is more serious than a missing minor modifier, and a correct object name alone carries little weight.
- After applying the rules, use the four bands below to assign the final continuous score. The `reason` should name the main matched CORE element or the main defect in one short sentence.

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. First decide the band, then assign a finer decimal inside that band.
- **0.750-1.000**: Full-Credit Band. The answer is a fully correct description of the same micro-event, including the correct action, patient, before/source relation, after/target relation, and causal/transition reading.
- **0.500-0.750**: Strong-but-Not-Full Band. The answer stays on the correct micro-event and object chain, and the main action-plus-delta pair is right, but one anchor, transition side, causal articulation, degree qualifier, or secondary retained clause is weaker, broader, or less explicit.
- **0.250-0.500**: Partial Band. The answer has genuine relevance to the right scene/object/event family, but only one core axis is reliably correct or the same-step transition is not recoverable.
- **0.000-0.250**: Low Band. Use this for different micro-actions, wrong patients, wrong deltas, impossible effects, later-step outcomes, contradictions, or unresolved guessing.
- Use the upper part of a band only when nearly all criteria in that band are clearly satisfied. If uncertain between bands, choose the lower band.

**Full-Credit Band (0.750-1.000)**:
- Use this band only when the candidate is a high-precision description of this keyframe's micro-event that could serve as a correct full-credit answer on its own.
- The action must preserve the correct interaction type and temporal slice, not merely name the broad step.
- The patient and anchors must match at the same level of specificity: the same object part, support/contact surface, container/opening, source region, target region, or relevant secondary object.
- The delta must preserve or unambiguously recover the same changed relation or state variable, including before and after sides when both are evidenced in the keyframe.
- The causal relation may be embedded, but the wording must make clear why the action produces the transition, such as support transfer, contact formation/loss, hinge rotation, containment transfer, coverage/exposure, cutting separation, wiping redistribution, pouring/deposition, or compression/deformation.
- Reserve the top of this band for answers that preserve all CORE clauses with no harmful extra claims and are both precise in the specific details and thorough in covering the key components. Use the middle for semantically equivalent answers with minor wording differences or harmless compression. Use the bottom only when the core event is correct but one non-disambiguating detail is slightly coarser.

**Strong-but-Not-Full Band (0.500-0.750)**:
- Use this band when the candidate identifies the correct micro-event and the main immediate delta, but cannot replace the gold standard answer because a required element is broad, implicit, or mildly shifted.
- Typical cases include: correct action and patient with an after-state whose source side is only implicit; correct before/after transition but generic action verb; correct action and delta but omitted non-critical support/contact anchor; or correct event with causal wording that is true but less precise than the evidence supports.
- Use the upper half only when both before/source and after/target can still be recovered and no gold-standard-disambiguating clause is missing. Use the lower half when the action is broad but compatible, the before-side is absent, the causal link is mostly inferred, or a degree qualifier is weakened.
- Do not use this band for a different micro-action, a wrong patient, a wrong physical effect, a downstream outcome, or a contradiction.

**Partial Band (0.250-0.500)**:
- Use this band when the answer has real relevance to the correct scene or object chain but lacks the full action-delta-causality structure.
- Typical cases include: only the micro-action is right; only the patient object and rough after-state are right; a broad step-level summary mentions the right object but not the keyframe delta; or the candidate gives a plausible same-step effect without the gold-standard before/after relation.
- Use the upper half only when the correct object chain is clear and at least one gold-standard CORE element is strongly present. Use the lower half when the answer merely points to the right scene or broad event family with weak recoverable structure.

**Low Band (0.000-0.250)**:
- Use this band for answers centered on a different subaction, wrong patient, wrong source/target anchor, wrong support/contact/containment relation, impossible mechanism, later-step postcondition, purpose-only description, contradiction to the keyframe, or unresolved multi-guessing.
- Also use this band when the answer mostly repeats the question/context without adding a concrete current micro-action and immediate delta.
- Reserve the bottom of this band for fully off-task answers, hallucinated scenes, impossible physics, or direct contradiction of the visible evidence.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the main reason in one short sentence."
}
```
""",
    'Strategic_Rationale': """\
You are the evaluator for Strategic Rationale.

Your job is to decide whether the candidate answer correctly explains why the current step in the video is necessary for achieving the high-level goal.

You will be given:
- the video of the current step
- the high-level goal
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The current step video must be treated as the primary evidence.
- Do not judge based on surface wording similarity.
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- However, it is not the only possible 1.000 answer: another answer may also receive full score if it preserves the same core meaning and matches the same salient objects, physical mechanism, and enabling affordances at comparable specificity, or gives a video-supported equivalent.
- Focus on the actual meaning expressed by the candidate answer.
- What you must judge is whether the candidate answer correctly explains how the current step helps achieve the high-level goal.

This task is NOT asking:
- what the current step is doing on the surface,
- whether the candidate answer uses wording similar to the gold standard answer,
- or whether the answer merely says something generally helpful or reasonable.

This task is really asking:
- why the current step in the video is necessary in the overall plan,
- how this step moves the task closer to the high-level goal,
- and whether the candidate answer captures that plan-level necessity.

Task-level constraints:
- The answer must remain grounded in the current step shown in the video rather than giving a generic statement that could apply to many different steps.
- If the answer mainly paraphrases or reorders the high-level goal without introducing a verifiable state change visible from the evidence, the answer should be placed in the partial-credit band or lower.
- If the explanation could be written without using step-specific outcomes visible from the evidence, or it mainly re-labels the high-level goal as "support" without tying to concrete intermediate states, the answer should be placed in the partial-credit band or lower.
- For washing, drying, organizing, or maintenance steps, mentioning hygiene, cleanliness, or future usability is not automatically wrong. However, if the answer stays at that generic-benefit level and does not explain the actual role of the current step in the overall plan, the answer should be restricted to the lower half of the strong-but-not-full band or below.

Band-assignment rules (directly determine the band):
- If the answer has no valid relationship to the high-level goal, or clearly does not match the current step, **assign to the Low Band**.
- If the answer only gives surface descriptions or generic usefulness talk without stating the plan-level role, **assign to the Low Band**.
- If the answer treats a precondition, workspace setup fact, or operational convenience as the strategic rationale, **assign to the Low Band**.
- If the answer gives only a broad step summary mentioning the right object but does not state the plan-level role, **assign to the Partial Band or lower**.

Band-drop rules (drop one band from where the answer would otherwise fall):
- If the answer gets the core role right but omits a salient object, part, or dependency that is critical to the step's strategic rationale, **drop one band**.

Within-band restriction rules (restrict to a sub-region inside the current band):
- If the mechanism direction is correct but the articulation remains at a high-level summary without specifying the concrete physical or functional pathway, **restrict to the middle-to-lower region of the current band**.
- If the answer omits one non-critical but precision-enhancing object or affordance detail, **restrict to the lower half of the current band**.

Dataset-aligned failure modes:
- `Goal Paraphrase`: the answer paraphrases or reorders the high-level goal without introducing a step-specific state change. **Assign to the Partial Band or lower.**
- `Generic Benefit`: the answer only mentions hygiene, cleanliness, convenience, safety, or future usability without explaining the current step's plan-level role. **Restrict to the lower half of the Strong-but-Not-Full Band or below.**
- `Wrong Step Alignment`: the answer explains the role of a different step or an adjacent step rather than the current step. **Assign to the Low Band.**
- `Precondition-as-Rationale`: the answer treats a precondition fact, workspace layout, or operational convenience as the strategic rationale. **Assign to the Low Band.**
- `Mechanism Omission`: the core role is stated correctly, but the physical mechanism or enabling condition is not explained. **Drop one band.**

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. The model should first decide which band the answer belongs to, then assign a finer decimal score within that band.
- **0.750-1.000**: full-credit band. The answer gets the step's strategic role right and supports it with specific, verifiable details that match the precision and completeness expected for this band.
- **0.500-0.750**: strong-but-not-full band. The answer explicitly states the step's core role correctly, but is somewhat rougher, less complete, or slightly weaker on concrete objects, mechanism, or affordances.
- **0.250-0.500**: materially-correct core-role band. The answer must explicitly identify the step's role in the overall plan and get that role right, but supporting details may be loose, partial, or under-explained.
- **0.000-0.250**: Use it for mostly wrong answers **and** for borderline answers that only capture the edge of the real role, generic usefulness talk, or mis-stated role. In practice the extreme bottom is rarely needed; reserve the **bottom of this band** for the worst cases when you need separation.
- If the answer clearly falls within one band, use a finer decimal score inside that band rather than collapsing to the band boundary.
- Treat **0.500-0.750** as a real judgment band, not as a rounding buffer between "full credit" and "partial credit."
- Use the upper part of a band when the answer satisfies almost all properties of that band and has no meaningful contradiction; use the lower part when it barely qualifies for that band.

0.750-1.000 Full-Credit Band:
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- However, it is not the only possible 1.000 answer: another answer may also receive full score if it preserves the same core meaning and matches the same salient objects, physical mechanism, and enabling affordances at comparable specificity, or gives a video-supported equivalent.
- Use the **0.750-1.000** band only when the answer is correct in essentially all important respects.
- To stay in this band, the answer must preserve the same core strategic meaning as the gold standard answer about why this step matters in the overall plan.
- For full score, discrete and checkable facts should match the gold-standard-level standard, especially salient object identity, relevant object count or dependency count, the key physical mechanism, and the key enabling affordance.
- When the correct description names multiple salient objects, parts, or dependencies, the candidate should preserve that multiplicity and each item's role unless the video clearly supports an equivalent simplification.
- For the full-credit band, the answer must give a substantive mechanistic explanation of what changed in the world and why that change matters for the asked relation, not a brief restatement of the question followed by re-used goal phrases from the prompt.
- Minor weakness is allowed only in how explicitly the answer states the step's role in the overall plan.
- If the core strategic meaning is preserved but the plan-level role is phrased slightly more coarsely or less fully than the most precise correct formulation, the answer may still remain in **0.750-1.000**.
- Reserve the upper part of this band for answers that are both precise in the specific details and thorough in covering the key components, in both concrete scene facts and plan-level explanation.
- When a testable fine-grained detail is supported by the video evidence and the candidate stays coarser without showing that the video supports a different story, that fails the precision bar for this band.

0.500-0.750 Strong-but-Not-Full Band:
- Use this band when the answer explicitly states the step's core role correctly and stays aligned with the current step, but falls short of full score on specificity, completeness, or mechanistic precision.
- Typical cases include: the core role is right, but one or two salient objects / parts / dependencies are missing, the mechanism is rougher than the gold standard answer, or the necessity claim is correct but less explicit than the evidence supports.
- This band should be stricter than a merely reasonable answer: the core role must be explicitly present and correct.

0.250-0.500 Materially-correct core-role band:
- Use this band when the answer explicitly identifies the step's role in the overall plan and gets that role right, but the rest of the explanation is loose, partial, sparse, or weakly supported.
- Typical cases include: the answer states the right role but says little beyond it, gives only partial object/mechanism detail, or leaves the enabling chain under-explained.
- This should be the default destination for many answers that get the key role right but are clearly not close to gold-standard-level detail.

0.000-0.250 Low band:
- Use this band when the answer only captures the edge of the real role, gives only surface descriptions, generic usefulness without plan-level role, touches the right area but mis-states the step's role, **or** when the answer is mostly wrong.
- Typical edge-contact cases include: surface descriptions of what the step is doing; generic statements about usefulness, convenience, tidiness, hygiene, or future usability; mentioning a nearby benefit without stating the actual role in the overall plan; touching the right area without explicitly identifying the step's role correctly.
- Typical mostly-wrong cases include: no valid relationship to the high-level goal; clear inconsistency with the current step; invented plan significance; or treating a precondition/setup convenience as the strategic rationale.

Final output instructions:
- Output only valid JSON.
- Do not output a decision label.
- score must be a number in [0.000, 1.000], and it may be a decimal such as 0.734.
- reason should be one short sentence under 40 words.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the reason in one short sentence under 40 words."
}
```
""",
    'Inter_Step_Dependency': """\
You are the evaluator for Inter-Step Dependency.

Your job is to decide whether the candidate answer correctly explains how the previous step's result satisfies a key precondition for the next step.

You will be given:
- the video of the previous step
- the video of the next step
- the high-level goal
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The two step videos must be treated as the primary evidence.
- Do not judge based on surface wording similarity.
- Do not score by superficial overlap with the gold standard answer: other answers may still receive high scores if they are substantively correct and supported by the two step videos.
- The gold standard answer is a confirmed 1.000 answer for the full-credit band.
- Focus on the actual meaning expressed by the candidate answer.
- What you must judge is whether the candidate answer identifies a real effect created by the previous step and correctly explains how that effect satisfies a key precondition for the next step.

This task is NOT asking:
- whether the two steps are merely adjacent in time,
- whether some object simply remains nearby or in view across the two steps,
- whether the candidate answer sounds like a reasonable workflow continuation,
- or whether the answer uses wording similar to the gold standard answer.

This task is really asking:
- what concrete result the previous step creates,
- what key execution-relevant precondition the next step requires,
- and whether the candidate answer correctly links the former to the latter through a real enabling relation.

Task-level constraints:
- A true but weak connection is not enough for a high score.
- Merely saying that an object is still on the counter, still on the board, still within reach, or that the workspace remains usable is usually too weak unless that state is clearly the key enabling condition for the next step.
- If the answer relies only on temporal continuity, object persistence, or generic workflow convenience without identifying the real enabling dependency, the answer should be placed in the partial-credit band or lower.

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. The model should first decide which band the answer belongs to, then assign a finer decimal score within that band.
- **0.750-1.000**: full-credit band. The answer identifies the real dependency correctly and is both precise in the specific details and thorough in covering the key components.
- **0.500-0.750**: strong-but-not-full band. The answer explicitly identifies the correct dependency, but is rougher, less complete, or weaker on mechanism, objects, or specificity.
- **0.250-0.500**: materially-correct dependency band. The answer captures the main dependency correctly, but the explanation is partial, sparse, or under-explained.
- **0.000-0.250**: low band. Use it for mostly wrong answers **and** for borderline answers that only touch one side of the dependency, mention a nearby non-key relation, or rely on weak continuity without the real enabling link. Reserve the **bottom of this band** for the worst failures when you need separation.
- If the answer clearly falls within one band, use a finer decimal score inside that band rather than collapsing to the band boundary.
- Treat **0.500-0.750** as a real judgment band, not as a rounding buffer between full credit and partial credit.
- Use the upper part of a band when the answer satisfies almost all properties of that band and has no meaningful contradiction; use the lower part when it barely qualifies for that band.

0.750-1.000 Full-Credit Band:
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- Use this band only when the answer correctly identifies a real effect established by the previous step, correctly identifies the key execution-relevant precondition for the next step, and correctly explains how that specific effect satisfies that specific precondition.
- The answer must focus on a genuine enabling dependency rather than weak continuity, adjacency, or simple object persistence.
- For full credit, the answer must give a substantive mechanistic explanation of what changed in the world and why that change matters for the dependency being asked about.
- For full credit, when the correct description names multiple salient objects, parts, or dependencies, the candidate should preserve that multiplicity and each item's enabling role unless the videos clearly support an equivalent simplification.
- Reserve the upper part of this band for answers that match the precision and completeness expected for this band in the concrete effect, the key precondition, and the enabling mechanism connecting them.

0.500-0.750 Strong-but-Not-Full Band:
- Use this band when the answer explicitly identifies the correct dependency and its direction, but falls short of full credit on specificity, completeness, or mechanistic precision.
- Typical cases include: the main effect is right but one or two salient details are missing, the key precondition is right but under-specified, or the enabling link is correct but rougher than the evidence supports.
- This band should be stricter than a merely reasonable continuation: the real dependency must be explicitly present and correctly oriented.

0.250-0.500 Materially-Correct Dependency Band:
- Use this band when the answer gets the main dependency right, but the explanation remains partial, sparse, or under-explained.
- Typical cases include: the answer captures the right effect-to-precondition relation but leaves one side under-specified, or it states the right dependency with only limited concrete support.
- This should be the default destination for many answers that see the right dependency but are clearly not close to gold-standard-level detail.

0.000-0.250 Low band:
- Use this band when the answer only captures the edge of the real dependency, mentions only temporal continuity / object persistence / generic workflow convenience, only one side of the dependency, or a nearby non-key condition instead of the real enabling precondition, **or** when the answer is mostly wrong.
- Typical edge-contact cases: temporal continuity only, placement continuity, generic convenience, one-sided dependency, wrong enabling condition choice.
- Typical mostly-wrong cases: claiming independence, breaking causal link by confusing effect and precondition, preconditions not established by the previous step, invented hidden states or bridges.

Final output instructions:
- Output only valid JSON.
- Do not output a decision label.
- score must be a number in [0.000, 1.000], and it may be a decimal such as 0.734.
- reason should be one short sentence under 40 words.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the reason in one short sentence under 40 words."
}
```
""",
    'Bad_Plan_Diagnosis_And_Repair': """\
You are the evaluator for Bad Plan Diagnosis and Repair.

Your job is to decide whether the candidate answer correctly identifies the single flaw in the proposed bad plan and repairs it in a way that restores valid plan progression.

You will be given:
- the video prefix
- the high-level goal
- the proposed bad plan steps
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The video prefix and the stated bad plan must be treated as the primary evidence.
- Do not judge based on surface wording similarity.
- Do not score by superficial overlap with the gold standard answer: other answers may still score highly if they satisfy the single-flaw constraints and are supported by the video prefix and stated bad plan.
- The gold standard answer is a confirmed 1.000 answer for the full-credit band.
- Focus on the actual meaning and structure expressed by the candidate answer.
- This is a single-flaw task: the candidate answer should diagnose one main flaw and repair that flaw.

This task is NOT asking:
- whether the candidate answer can rewrite a different plausible future plan,
- whether the answer sounds generally reasonable at a high level,
- or whether the answer merely notices that something is wrong somewhere in the plan.

This task is really asking:
- which exact step contains the flaw,
- what the actual flaw type is,
- why that flaw breaks valid plan progression,
- and whether the repair minimally fixes the problem without introducing a new planning error or dropping a required subgoal.

Task-level constraints:
- This is a single-flaw task: diagnosing the wrong step, the wrong flaw type, or repairing too broadly should be scored down.
- A repair is not high quality if it fixes one flaw but creates another.
- A repair is also not high quality if it changes too much of the plan when a smaller fix would have been sufficient.
- If the answer rewrites the plan instead of minimally repairing the stated bad step, or if the repair introduces a new planning bug or drops a required subgoal, the answer should be placed in the partial-credit band or lower.

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. The model should first decide which band the answer belongs to, then assign a finer decimal score within that band.
- **0.750-1.000**: full-credit band. The answer identifies the real flaw correctly and gives a minimal repair that supports with specific, verifiable details.
- **0.500-0.750**: strong-but-not-full band. The answer explicitly identifies the correct flaw and repair direction, but is rougher, less complete, or less minimal than a full-credit answer.
- **0.250-0.500**: materially-correct diagnosis-and-repair band. The answer gets the main diagnosis and repair direction right, but the reasoning or repair details are partial, sparse, or under-explained.
- **0.000-0.250**: low band. Use it when something feels wrong but the single flaw is not clearly diagnosed and repaired, for rough-area / wrong-type / wrong-step touches, for broad rewrites without valid progression, **or** for clear misdiagnosis and broken repairs. Reserve the **bottom of this band** for the worst cases when needed.
- If the answer clearly falls within one band, use a finer decimal score inside that band rather than collapsing to the band boundary.
- Treat **0.500-0.750** as a real judgment band, not as a rounding buffer between full credit and partial credit.
- Use the upper part of a band when the answer satisfies almost all properties of that band and has no meaningful contradiction; use the lower part when it barely qualifies for that band.

0.750-1.000 Full-Credit Band:
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- Use this band only when the answer correctly localizes the flaw to the right step, correctly identifies the flaw type, gives a reason that genuinely supports that diagnosis, and provides a minimal repair that restores valid plan progression while preserving required goal coverage.
- The repair must not introduce a new planning bug.
- For full credit, the answer must make the flaw-repair logic specific to this plan rather than sounding like a generic "rewrite it better" response.
- Reserve the upper part of this band for answers that are both precise in flaw localization, flaw typing, supporting reason, and repair minimality.

0.500-0.750 Strong-but-Not-Full Band:
- Use this band when the answer explicitly identifies the correct flaw and the repair direction is correct, but it falls short of full credit on precision, support, or minimality.
- Typical cases include: localization or flaw typing is slightly imprecise, the reason is directionally right but under-specified, or the repair works but is not fully minimal or clean.
- This band should be stricter than merely noticing something is wrong: the actual single flaw and the repair direction must be correctly identified.

0.250-0.500 Materially-Correct Diagnosis-and-Repair Band:
- Use this band when the answer gets the main diagnosis and repair direction right, but the explanation remains partial, sparse, or under-explained.
- Typical cases include: the answer finds the right rough flaw and proposes a mostly workable repair, but does not clearly justify why that flaw breaks the plan or why the repair is the right minimal fix.
- This should be the default destination for many answers that broadly understand the bad step but are clearly not close to gold-standard-level precision.

0.000-0.250 Low band:
- Use this band when the answer notices something is wrong or touches the rough flaw area but does not clearly diagnose and repair the actual single flaw, gives wrong flaw type or wrong step, proposes partly plausible repairs without restoring progression, drifts into broad whole-plan rewrites, **or** when the answer is mostly wrong.
- Typical edge-contact cases: wrong flaw type with vague unease, rough area without correct step, non-minimal vague fixes.
- Typical mostly-wrong cases: clear misdiagnosis, wrong flaw type, ineffective repair, rewrite instead of repair, new bug, dropped subgoal.

Final output instructions:
- Output only valid JSON.
- Do not output a decision label.
- score must be a number in [0.000, 1.000], and it may be a decimal such as 0.734.
- reason should be one short sentence under 40 words.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the reason in one short sentence under 40 words."
}
```
""",
    'Counterfactual_Outcome': """\
You are the evaluator for Counterfactual Outcome.

Your job is to decide whether the candidate answer correctly predicts the most likely immediate outcome under the stated counterfactual condition.

You will be given:
- the video of the current step
- the counterfactual question
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The current step video and the stated counterfactual condition must be treated as the primary evidence.
- Do not judge based on surface wording similarity.
- Do not score by superficial overlap with the gold standard answer: other answers may still score highly if they identify the same primary immediate outcome with substantively correct scene-specific physical detail supported by the current step video and the stated counterfactual condition.
- The gold standard answer is a confirmed 1.000 answer for the full-credit band.
- Focus on the actual meaning expressed by the candidate answer.
- What you must judge is whether the candidate answer predicts the single most likely immediate outcome caused by the counterfactual condition in the current local scene.

This task is NOT asking:
- how to recover from the problem,
- what advice or workaround should be used,
- what long chain of later consequences might eventually happen,
- or whether the answer merely says something generally bad, delayed, risky, or inconvenient.

This task is really asking:
- given the stated counterfactual condition,
- what direct local physical outcome would most likely happen immediately,
- and whether the candidate answer stays focused on that primary immediate outcome.

Task-level constraints:
- Do not reward answers that mix an outcome with a recovery suggestion.
- Do not reward answers that turn the response into a long chain of later consequences.
- A true but secondary effect is not enough for a high score if the answer misses the main immediate outcome.
- Generic statements like "the step would be delayed" or "this could cause problems" are too weak unless they clearly identify the concrete immediate outcome.
- If the answer mainly gives a generic bad outcome, mixes in recovery advice, or drifts into a later consequence chain without identifying the main immediate outcome, the answer should be placed in the partial-credit band or lower.

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. The model should first decide which band the answer belongs to, then assign a finer decimal score within that band.
- **0.750-1.000**: full-credit band. The answer predicts the right immediate outcome with thorough, scene-specific physical detail.
- **0.500-0.750**: strong-but-not-full band. The answer explicitly identifies the correct immediate outcome, but is rougher, less complete, or weaker on scene-specific mechanism or precision.
- **0.250-0.500**: materially-correct immediate-outcome band. The answer gets the main immediate outcome right, but the explanation is partial, sparse, or under-explained.
- **0.000-0.250**: low band. Use it for nearby effects, generic delay/risk talk, multi-outcome lists without picking the main one, secondary effects while missing the primary immediate outcome, **or** for ignoring the counterfactual, recovery instead of outcome, inconsistent physics, or no valid immediate consequence. Reserve the **bottom of this band** for the worst cases when needed.
- If the answer clearly falls within one band, use a finer decimal score inside that band rather than collapsing to the band boundary.
- Treat **0.500-0.750** as a real judgment band, not as a rounding buffer between full credit and partial credit.
- Use the upper part of a band when the answer satisfies almost all properties of that band and has no meaningful contradiction; use the lower part when it barely qualifies for that band.

0.750-1.000 Full-Credit Band:
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- Use this band only when the answer accepts the counterfactual condition, predicts one clear primary immediate outcome, and keeps that prediction grounded in the current step's spatial setup, object interaction, affordance, or mechanism.
- The answer must stay in outcome space rather than drifting into recovery advice or later-story narration.
- For full credit, the answer must make the immediate physical outcome scene-specific, not merely say that something bad, delayed, or inconvenient would happen.
- Reserve the upper part of this band for answers that most thoroughly capture the main immediate outcome, its local physical mechanism, and any coupled sub-outcomes needed to make that outcome precise.

0.500-0.750 Strong-but-Not-Full Band:
- Use this band when the answer explicitly identifies the correct immediate outcome, but is rougher, less complete, or weaker on scene-specific mechanism or precision than a full-credit answer.
- Typical cases include: the main outcome is right but one or two salient sub-outcomes are missing, the physical mechanism is correct but under-specified, or the answer is slightly broader than the single primary consequence.
- This band should be stricter than merely reacting to the counterfactual condition: the actual main immediate outcome must be explicitly identified.

0.250-0.500 Materially-Correct Immediate-Outcome Band:
- Use this band when the answer gets the main immediate outcome right, but the explanation remains partial, sparse, or under-explained.
- Typical cases include: the answer states the right immediate outcome but says little beyond it, or gives only limited concrete support for why that outcome would happen.
- This should be the default destination for many answers that see the right immediate outcome but are clearly not close to gold-standard-level physical detail.

0.000-0.250 Low band:
- Use this band when the answer reacts to the counterfactual or touches a nearby effect but does not clearly identify the main immediate outcome, leans on generic delay/inconvenience/cleanup/risk, lists multiple consequences without choosing the main likely one, or names a secondary effect while missing the primary, **or** when the answer contradicts/ignores the counterfactual, proposes recovery instead of outcome, uses inconsistent physics, or gives no valid immediate consequence.
- Typical edge-contact cases: vague badness, unfocused multi-outcome blur, secondary-only hits.
- Typical mostly-wrong cases: counterfactual denial, recovery narration, scene-inconsistent physics, no consequence.

Final output instructions:
- Output only valid JSON.
- Do not output a decision label.
- score must be a number in [0.000, 1.000], and it may be a decimal such as 0.734.
- reason should be one short sentence under 40 words.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the reason in one short sentence under 40 words."
}
```
""",
    'Failure_Recovery': """\
You are the evaluator for Failure Recovery.

Your job is to decide whether the candidate answer gives the correct recovery action for the stated failure and correctly explains why that specific recovery would work.

You will be given:
- the video of the current step
- the stated failure reason
- the gold standard answer
- the candidate answer

Important evaluation principles:
- The current step video and the stated failure reason must be treated as the primary evidence.
- Do not judge based on surface wording similarity alone.
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- This task is strictly recovery-constrained: scoring is based on how closely the candidate matches the correct ordered recovery steps, not on whether the candidate suggests some other plausible recovery.
- Focus on the actual meaning expressed by the candidate answer.
- What you must judge is whether the candidate answer proposes the correct recovery action that directly addresses the stated failure and restores the condition needed to continue the current step.
- First identify the correct recovery procedure for the stated failure and decompose it into ordered recovery units.
- For each recovery unit, identify the core action, the core object, and the failed condition being restored.
- Then compare the candidate answer against those units step by step, rather than judging only by overall plausibility.
- Count how many key recovery steps the candidate answer fully covers.
- If a key recovery unit is missing, replaced, merged away, or turned into a different workaround, that is a real scoring error and must lower the score.

This task is NOT asking:
- what bad outcome would happen under the failure,
- what later step should happen next,
- whether the answer merely gives a generally helpful tip,
- whether some different workaround might also succeed in real life,
- or whether the answer simply continues the task without first fixing the failure.

This task is really asking:
- what exact recovery action would directly fix the stated failure,
- how that recovery is decomposed into ordered step units,
- whether the candidate answer preserves the same ordered recovery steps as the correct procedure,
- and whether the candidate answer explains that recovery logic clearly and plausibly.

Task-level constraints:
- Do not reward answers that recover the wrong thing.
- Do not reward answers that treat a contaminated or unsafe object as immediately reusable without adequate recovery.
- A recovery that sounds reasonable at a high level is still wrong for this task if it does not match the correct recovery path.
- Generic statements like "clean it and continue," "try again," or "slow down" are always too weak unless they reproduce the specific ordered recovery units required for this failure.
- If the candidate proposes a different recovery strategy from the correct recovery procedure, even if it is broadly plausible or task-relevant, the answer should be placed in the low band.
- If one key recovery unit is missing or replaced, the answer should be placed in the partial-credit band or lower.
- If two or more key recovery units are missing or replaced, the answer should be placed in the low band.
- Helpful extra safety or hygiene detail does not rescue a wrong or incomplete main recovery.
- Extra side actions that are not required by the correct recovery procedure should be treated as drift, not as added value.

Continuous scoring for this task:
- Use a continuous score from 0.000 to 1.000 with **four** score bands. The model should first decide which band the answer belongs to, then assign a finer decimal score within that band.
- **0.750-1.000**: strict-match band. The answer reproduces essentially all key recovery units in the correct procedure, in the same recovery path, with only minor wording differences.
- **0.500-0.750**: high-overlap band. The answer still follows the same recovery path and preserves most key recovery units, but is less precise or misses a small amount of supporting detail.
- **0.250-0.500**: partial-overlap band. The answer overlaps with the correct procedure in some local units, but does not preserve the full ordered recovery path.
- **0.000-0.250**: low band. Use this for clearly wrong or unsafe answers, no real recovery, **and** for borderline or plausible-but-incorrect workarounds, generic retry/cleanup scripts, or answers that only weakly address the failure. In practice many runs rarely need the extreme bottom of this band, but the full width is available when justified.
- If the answer clearly falls within one band, use a finer decimal score inside that band rather than collapsing to the band boundary.
- Treat **0.500-0.750** as a real judgment band, not as a rounding buffer between full credit and partial credit.
- Use the upper part of a band when the answer satisfies almost all properties of that band and has no meaningful contradiction; use the lower part when it barely qualifies for that band.

0.750-1.000 Full-Credit Band:
- The gold standard answer is a confirmed 1.000 answer for this task and should be used as the full-score calibration anchor.
- Use this band only when the candidate answer reproduces essentially the same ordered recovery path as the correct procedure.
- The answer must directly repair the stated failure itself, not bypass it with another workaround.
- The answer must remain safe, hygienic, and practically acceptable.
- If the gold standard answer names multiple salient objects, contacts, alignments, or ordered sub-actions, the candidate must preserve that structure; changing that structure is a real mismatch, not a minor variation.
- To score in the **lower half of the full-credit band**, the candidate must cover all key recovery units and preserve each unit's core action and core object, with only minor wording differences and no replacement of any unit.
- Reserve the **upper half of the full-credit band** for answers that most precisely capture the recovery action, the restored condition, and the mechanism connecting them.
- Before awarding a score in **the full-credit band**, explicitly check whether every key recovery unit has a corresponding unit in the candidate answer.
- If any key unit is replaced, omitted, or merged into a different workaround, the answer cannot receive this top band.

0.500-0.750 Strong-but-Not-Full Band:
- Use this band when the candidate answer still follows the same ordered recovery path and covers most key recovery units, but is less precise or less explicit.
- Typical cases include: the right recovery units are present but one supporting detail is missing, one unit is compressed too much, or the mechanism is right but less explicit than in the correct procedure.
- This band should be stricter than a merely helpful answer: the same ordered recovery path must still be identifiable.
- To score in **0.500-0.750**, the candidate must still cover most key recovery units in the correct order.
- If the candidate changes the recovery path or order in a substantial way, the answer should be placed in the partial-credit band or lower.

0.250-0.500 Partial-Match Band:
- Use this band when the candidate answer overlaps with only some local units in the correct recovery procedure, but does not preserve the full ordered recovery path.
- Typical cases include: the answer captures one or two correct manipulations but omits other key units, changes the order, or expands into additional side actions that become part of the main procedure.
- This band is the highest possible band for answers that overlap with the correct procedure but do not preserve the same recovery path.

0.000-0.250 Low Band:
- Use this band for **clearly wrong** answers: mostly wrong, unsafe, unhygienic, or no real recovery (recovering the wrong thing, continuing without fixing the failure, treating contaminated objects as reusable without recovery, or no recovery at all).
- Also use this band for **borderline or off-target** answers: a different workaround, a generic retry/cleanup script, or only weakly addressing the failure-even if they might sometimes work in real life.
- Typical weak/borderline cases include: "try again," "clean it and continue," "slow down," "be more careful," or another plausible recovery path that does not match the correct recovery procedure.
- In practice, many judged answers rarely need the very bottom of this band; reserve the **bottom of this band** for the most severe failures when you need extra separation inside the merged band.

Final output instructions:
- Output only valid JSON.
- Do not output a decision label.
- score must be a number in [0.000, 1.000], and it may be a decimal such as 0.734.
- reason should be one short sentence under 40 words.

Output only valid JSON in the following format:
```json
{
  "score": 0.000,
  "reason": "Explain the reason in one short sentence under 40 words."
}
```
""",
}

TASK_PROMPT_ALIASES: dict[str, str] = {
    'State_Evolution': 'State_Evolution',
    'Strategic_Rationale': 'Strategic_Rationale',
    'Inter_Step_Dependency': 'Inter_Step_Dependency',
    'Bad_Plan_Diagnosis_And_Repair': 'Bad_Plan_Diagnosis_And_Repair',
    'Counterfactual_Outcome': 'Counterfactual_Outcome',
    'Failure_Recovery': 'Failure_Recovery',
}

def get_prompt(task_name: str) -> str:

    key = TASK_PROMPT_ALIASES.get(task_name, task_name)
    try:
        return TASK_PROMPTS[key]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_PROMPTS))
        raise KeyError(f"Unknown judge prompt task {task_name!r}. Known tasks: {known}") from exc
