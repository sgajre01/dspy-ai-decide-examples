# Blog research: DSPy Predict, business policies, and ReAnchor with ai_decide

Updated October 25, 2026. Research notes and writing handoff; **not a blog draft**.

## Current conclusion

**Predict and actual ReAnchor work with live Databricks ai_decide binary probabilities through our small custom client. Compatibility is proven; useful optimizer gains are not.**

The full email example passed end to end. ReAnchor retained the ordinary 0.5 threshold and matched the baseline on the held-out test emails. An earlier, separate controlled test changed a deliberately poor threshold from 1.0 to 0.6, establishing that parameter fitting works without establishing practical benefit.

## Read these notes in order

1. [Experiment findings](EXPERIMENT_RESULTS.md): what ran, exact results, provenance, and limitations.
2. [Code examples](CODE_EXAMPLES.md): excerpts using the working companion client and pinned environment.
3. [Article plan](ARTICLE_PLAN.md): the argument, teaching sequence, and unresolved editorial decisions.
4. [Complete run evidence](../../results/README.md): executed snapshot, raw responses, predictions, and saved policy.

The [project README](../../README.md) explains setup. The [source notebook](../../notebooks/email_triage.ipynb) is the runnable tutorial. The [private repository](https://github.com/sgajre01/dspy-ai-decide-examples) is the current sharing destination; public release has not been requested.

## The three responsibilities

| Responsibility | What it means in the example |
| --- | --- |
| Task definition | Signature inputs, descriptions, and instructions define whether an email needs a reply. |
| Model assessment | ai_decide estimates the probability. The custom client preserves it. |
| Business policy | DSPy compares that probability with an application-selected threshold to produce a boolean. |

Predict provides readability, reuse, and separation of a task from its backend. A compatible provider can potentially replace the backend while preserving the input/output contract, but integration work, evaluation, and threshold tuning may still be needed. We have not executed a provider swap.

**You can set that policy manually or dynamically, based on business context.** In the example, application logic chooses between configured predictors for critical and standard customers. This lets the same model assessment lead to different actions under different policies. The application supplies the context; DSPy does not discover the business rule by itself.

**ReAnchor typically runs during development or periodic tuning. The application can then use the fitted threshold—or select among separately evaluated policies dynamically based on business context. ReAnchor does not automatically infer that context or the application's business priorities.**

The live example tested selection between fixed illustrative contextual policies. Separately fitting policies for different contexts remains a possible extension, not a completed experiment.

## Claims we can make

| Claim | Evidence and qualification |
| --- | --- |
| Predict can call ai_decide | Live ticket and email experiments using our custom synchronous binary client. |
| Original probabilities are available to application code | Explicit `Noul` exposes `.probability`; the client-to-prediction mapping was checked. |
| DSPy can apply business thresholds | Manual thresholds and selection of contextual policies passed in the email notebook. |
| Actual ReAnchor can consume the evidence | Unmodified installed optimizer compiled the predictors successfully. |
| ReAnchor can change a threshold | Earlier controlled 1.0 → 0.6 check on eight synthetic tickets. Deliberately poor baseline; not a benefit result. |
| ReAnchor did not improve this email example | Retained 0.5; same zero missed replies and one false alert as baseline on 18 test emails. |
| Cached evidence avoided repeat inference during this compile | Zero additional logical backend calls during the email compile after training/validation inference. Single-predictor observation, not a universal call-count guarantee. |
| Predict can help readability, reuse, and backend separation | An architectural rationale illustrated by the code, not a measured productivity or quality gain. |
| Jev has a ready-made client in the tested DSPy release | Supported by the inspiration article and inspection of DSPy 3.4.1; no Jev service was called in our experiments. |

## Claims we cannot make

- That ReAnchor improved ai_decide accuracy or business outcomes on these emails.
- That ai_decide requires switching to Jev to obtain ReAnchor support.
- That our client exactly reproduces every capability of the Jev integration or is an official/native connector.
- That Predict, threshold fitting, or ReAnchor retrains ai_decide, changes returned probabilities, or makes probabilities better calibrated.
- That Score cuts, Choice weights, plain bool/Literal outputs, async, GEPA, or arbitrary provider interchangeability were tested through our client.
- That our synthetic fixture establishes real-inbox quality, financial savings, faster inference, or provider superiority.
- That the earlier GENAI availability rejection proves an announced Free Edition withdrawal.

## Jev guidance for the eventual article

The author prefers starting with ai_decide for decision work. That preference is distinct from a proven universal ranking of models. The binary result removes exclusive ReAnchor access as a reason to switch to Jev.

Considering Jev can still be reasonable when a team values its ready-made DSPy integration more than maintaining a custom client tied to an experimental interface. That is an integration tradeoff. Our work does not establish Jev's relative accuracy, price, speed, governance fit, or operational suitability for a particular reader.

## Version and terminology check

Verified for these notes on October 25, 2026:

- [ai_decide](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_decide) is **Beta**; our REST requests selected function version 1.0.
- [ReAnchor](https://dspy.ai/current/api/experimental/ReAnchor/) and the relevant DSPy decision interfaces are **experimental**.
- Execution used DSPy **3.4.1**, Databricks SDK **0.157.0**, and Python **3.10.12** on Azure Databricks.
- DSPy 3.4.1 includes `dspy.experimental.TypeSafe`. Current development documentation describes a newer client interface; use the installed release and executed evidence for version-specific code claims.

Inspiration: [Building & Optimizing Jev Programs with DSPy](https://www.cmpnd.ai/blog/building-jev-programs-with-dspy.html), Drew Breunig and Isaac Miller. Its reported optimization gains belong to their workload and cannot be presented as our results.
