# Article plan based on the completed experiments

Updated October 25, 2026. This is an outline and editorial handoff, not article prose.

## Reader and purpose

Write for a Databricks practitioner building binary decisions such as email triage, classification, or routing. The reader should understand how Predict can organize an ai_decide-backed task, how application policies turn probabilities into actions, and what actual ReAnchor integration does and does not establish.

The author's starting preference is ai_decide for decision-type work. Preserve that preference while separating it from an unproven claim of universal model superiority.

## Thesis

**A small custom client lets DSPy Predict and actual ReAnchor work with Databricks ai_decide binary probabilities. DSPy helps separate tasks, backends, and decision policies; optimizer benefit still depends on the workload and must be measured.**

The strongest empirical conclusion is compatibility, with no optimizer improvement in our full email example. Do not frame the article as a demonstrated accuracy success or as a reason readers must switch to Jev.

Working title: **DSPy Predict and ReAnchor with Databricks ai_decide: What Works, and What Actually Improves?**

## Proposed teaching sequence

1. **Lead with the result.** State the custom-client binary compatibility result and the absence of an email quality gain. Establish the original Jev article as the inspiration.
2. **Define the task.** Use the same synthetic email triage example throughout. Explain what counts as a reply and how thread context resolves responsibility. Disclose that it is synthetic.
3. **Show Predict using ai_decide.** Present the signature and short application call, then reveal the small custom response translation. Explain readability, reuse, and backend separation without promising plug-and-play providers or automatic quality gains.
4. **Separate probability from policy.** Use a clearly hypothetical probability 0.40 at thresholds 0.50 and 0.30. Then connect it to the actual `.probability` and `.value` interface. Explain the false-negative/false-positive tradeoff.
5. **Introduce business context.** Show application selection between configured policies. Explain that task descriptions define what is assessed while thresholds define how much evidence is needed to act. Context and priorities are supplied explicitly.
6. **Explain ReAnchor's role.** It fits thresholds from labeled examples and a supplied metric during development or periodic tuning. Keep model training, probability calibration, and prompt rewriting separate from this mechanism.
7. **Show the real optimizer call and evaluation.** Declare the 2:1 example costs and 36/18/18 splits. Show actual `ReAnchor.compile()`, unchanged threshold 0.5, and all final test error counts. Explain why retaining the original policy is legitimate.
8. **Distinguish capability from benefit.** Mention the earlier deliberately poor 1.0 → 0.6 smoke check only as parameter-fitting evidence. State the cache observation separately from latency, billing, or useful quality claims.
9. **Give a scoped recommendation.** Use direct ai_decide calls where sufficient. Use Predict when its organization/abstraction helps; consider ReAnchor when labeled data and a business metric justify tuning. Consider Jev for its ready-made integration where relevant, rather than exclusive optimizer access.
10. **End with reproducibility and boundaries.** Link the repository, source notebook, executed snapshot, dataset notes, and result JSON. State pinned versions, feature maturity, custom-client scope, and which additional evidence would justify a benefit claim later.

## Required distinctions

- **Backend abstraction:** task and application contract can be reused with a compatible backend; a real provider swap was not tested.
- **Policy abstraction:** the application can change actions while preserving model probabilities. Direct SQL/Python can also apply thresholds.
- **Manual/dynamic selection:** the application chooses a policy using supplied context.
- **ReAnchor fitting:** labeled data and a metric determine whether a candidate numeric setting is retained. ReAnchor does not infer the business objective or runtime context.
- **Controlled check versus evaluation:** intentionally poor threshold fitting proves capability; ordinary-baseline held-out evaluation measures benefit.
- **No gain versus failure:** retaining 0.5 is compatible execution with no observed gain, not an integration failure.
- **Jev integration versus comparison:** a ready-made client is documented; no Jev inference or quality/cost/speed benchmark was run by us.

## Evidence to include

Use the exact test table in [experiment findings](EXPERIMENT_RESULTS.md). Keep the logical call counts and cached compilation observation scoped to this run. Cite the original article for its own claims without reusing its improvements as ours. Prefer the repository's inspected pinned source and live evidence for version-specific behavior.

Include short code excerpts from [CODE_EXAMPLES.md](CODE_EXAMPLES.md), with links to the full implementation. The full signature's instructions matter to the results; disclose when a displayed excerpt omits them for readability.

## What should stay out of the main narrative

The Free Edition troubleshooting and initial development-interface mismatch can be brief setup notes if useful. They should not displace the core probability/policy lesson or imply a product-wide shutdown. A robust connector, extra decision types, production deployment pipeline, and a broad vendor comparison are outside the agreed scope.

## Remaining decisions and future research

Publication destination and final length remain undecided. Public release of the currently private repository has not been requested. Before publication, recheck official documentation, maturity labels, package interfaces, notebook links, and the selected code commit or release reference.

A meaningful optimization-benefit claim would need a representative workload, clearly justified costs, and a new protected final evaluation. That is optional future research; do not manufacture a gain by modifying this test set after seeing its outcomes. No further experiment or full blog draft is part of the current documentation task.
