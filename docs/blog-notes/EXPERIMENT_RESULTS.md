# Experiment findings for the blog

Verified October 25, 2026. The experiments answer compatibility and policy questions; they do not establish real-world optimization benefit.

## Verdict by question

| Question | Verdict |
| --- | --- |
| Can Predict use Databricks ai_decide? | Yes, through our custom client for synchronous binary Noul output. |
| Can the actual ReAnchor optimizer run on those probabilities? | Yes. No optimizer, predictor, or decoder patch was required. |
| Can it actually change a setting? | Yes, in an earlier controlled threshold test. |
| Did it improve the default baseline in the email example? | No; it retained 0.5 and produced the same test outcomes. |
| Is this native/full Jev-equivalent integration? | Not established; a custom, version-specific client was required. |

## 1. Initial compatibility proof: eight synthetic tickets

The original Azure notebook used six training tickets and two validation tickets. The ordinary 0.5 threshold already classified those simple examples correctly, so ReAnchor retained the policy.

A separate controlled check started with an intentionally poor threshold of 1.0. Actual ReAnchor fitted 0.6 and passed its fold check. The eight live Databricks responses were reused from cache; compilation and the stress check made no additional calls.

This establishes functioning parameter fitting. The deliberately poor starting point is not an ordinary baseline or an accuracy benchmark. There was no separate final test split. Save/load was checked for the default-policy predictor; persistence of the changed 0.6 stress policy was not separately tested.

- [Original notebook](https://adb-8619709329887883.3.azuredatabricks.net/editor/notebooks/4012772188116690?o=8619709329887883)
- [Original successful run](https://adb-8619709329887883.3.azuredatabricks.net/?o=8619709329887883#job/780381782339134/run/1016160674179100)

## 2. Full email tutorial: 72 synthetic examples

The hand-authored dataset and its labels/split membership were fixed before inference. Inputs include sender, subject, body, and thread history. Labels, rationales, IDs, split names, and customer tier are excluded from model inputs. The class balance and repeated task patterns make this a teaching fixture, not a representative inbox benchmark. See [data provenance and labeling rules](../../data/README.md).

| Split | Emails | Positive labels | Negative labels | Role |
| --- | ---: | ---: | ---: | --- |
| Training | 36 | 18 | 18 | Fit numeric policy settings. |
| Validation | 18 | 9 | 9 | Before/after reporting; the installed optimizer does not fit on these examples. |
| Final test | 18 | 9 | 9 | Evaluate frozen policies after fitting. |

The signature asks whether the recipient needs to reply to an outstanding request. Direct and indirect requests assigned to the recipient are positive. Informational, automated, optional, resolved, and other-owner requests are negative, with thread history resolving responsibility.

The declared business costs were 2 for a missed reply and 1 for an unnecessary alert. Correct decisions cost zero. The metric maximizes mean `1 - error_cost / 2`, equivalent to minimizing mean weighted error cost. These are illustrative configurable costs, not measured business economics.

### What actual ReAnchor did

- Started at the ordinary threshold 0.5.
- Recorded 36 training observations, with 16 distinct probabilities ranging from 0.0 to 0.99.
- Reported 17 candidate settings.
- Found the starting policy already scored 1.0 on training and validation.
- Retained the original setting; the report says `fitted behavior did not beat the original`.
- Recorded zero passed and zero failed fold checks in this run. Do not describe an accepted fitted candidate passing a fold check here; that belongs to the earlier controlled ticket check.

### Final held-out results

| Frozen policy | Threshold | Missed replies | False alerts | Correct / 18 | Total weighted error cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 0.5 | 0 | 1 | 17 | 1 |
| Manual | 0.3 | 0 | 2 | 16 | 2 |
| ReAnchor | Retained 0.5 | 0 | 1 | 17 | 1 |
| Contextual | Critical 0.3; standard 0.5 | 0 | 1 | 17 | 1 |

**No optimizer quality improvement was observed.** The manual lower threshold added a false alert without catching an additional missed reply. The contextual policy matched the baseline's aggregate error counts; that does not prove it is beneficial generally. Test results were not used to pick a new threshold, relabel emails, revise costs, or change split membership.

### Integration checks that passed

Predict preserved the raw model probability. Manual policy changes reused cached evidence. Baseline and optimized test probabilities matched. Application selection between configured contextual predictors matched explicit probability/threshold comparisons. Saving and restoring the resulting 0.5 policy reproduced predictions. The installed ReAnchor source hash stayed unchanged.

Six focused offline checks separately verified probability preservation, an inclusive threshold boundary, valid zero probabilities, invalid-probability rejection, metric arithmetic, contextual policy decisions, and fixture/input contracts. These checks use fake responses for plumbing; only the live notebook supplies model evidence.

### Calls and timing

The run made 72 logical backend calls, one per unique email. Training and validation were already cached when compilation began, and compilation added zero calls. This observation is specific to this fixed-input, single-predictor program. Changing upstream decisions in a composed program can create new requests.

Recorded compilation wall time was approximately 0.34 seconds in this run, after inference had populated the cache. It is diagnostic timing, not a repeated latency benchmark. Logical SDK calls do not separately count transport retries and cannot establish billed cost or financial savings.

## Reproducibility and evidence

| Item | Recorded value |
| --- | --- |
| Platform | Azure Databricks, serverless notebook run |
| Python | 3.10.12 |
| DSPy | 3.4.1 |
| Databricks SDK | 0.157.0 |
| ai_decide function version | 1.0 |
| Dataset SHA-256 | `a36d5a95243170a2b79a5b0c009e6c145f4da11243715ae60e913e8870f934a0` |
| ReAnchor source SHA-256 | `33fa0d10d9ba228b40d6ac066987bc5ac4d6a972e07544c32e9c7ce8a20908d8` |

- [Run summary and source hashes](../../results/initial_run/evaluation_summary.json)
- [Predictions for every evaluated policy](../../results/initial_run/predictions.json)
- [Original Databricks responses](../../results/initial_run/live_call_evidence.json)
- [Saved policy](../../results/initial_run/optimized_policy.json)
- [Executed notebook snapshot](../../results/initial_run/email_triage.executed.ipynb)
- [Source notebook](../../notebooks/email_triage.ipynb)
- [Live email notebook](https://adb-8619709329887883.3.azuredatabricks.net/editor/notebooks/3115261995240150?o=8619709329887883)
- [Successful email run](https://adb-8619709329887883.3.azuredatabricks.net/?o=8619709329887883#job/98257362090357/run/515278235904014)

The executed snapshot contains outputs extracted from the actual run export, not a replay. Package-install logs were omitted, the structured exit payload is stored as adjacent JSON, and native Jupyter execution counts were unavailable. Workspace links require workspace access.

## Earlier availability and implementation failures

GENAI's current requests failed despite stored successful ai_decide outputs from the user's earlier Land or Water work. That workspace-specific rejection did not establish ReAnchor incompatibility or a published Free Edition withdrawal. Azure initially reported the preview disabled; the user enabled it and subsequent calls succeeded.

An initial adapter attempt followed the newer development interface rather than the installed DSPy release and failed before any Databricks inference call. Correcting the client to the actual 3.4.1 interface produced the successful result. Keep these failures as setup/version lessons rather than evidence against compatibility.

## Remaining evidence gaps

No representative inbox evaluation, useful held-out optimizer gain, Jev quality/cost/speed comparison, live backend swap, changed-threshold persistence check, separately fitted contextual policies, or custom-client coverage of plain bool, Literal, Score, Choice, async, or GEPA has been established. Future benefit claims need a newly designed evaluation with representative labels, declared costs, and untouched final test data.
