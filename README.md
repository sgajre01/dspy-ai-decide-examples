# DSPy Predict and ReAnchor with Databricks ai_decide

A small runnable email-triage example: connect DSPy `Predict` to Databricks
`ai_decide`, inspect probabilities, apply explicit business policies, and fit a
binary threshold with the actual ReAnchor optimizer. This is companion code in
development for an article, not the article itself.

The [blog research notes](docs/blog-notes/README.md) collect the verified findings,
experiment boundaries, code excerpts, and article outline. The integration works;
the completed email run showed no optimizer quality gain over threshold 0.5.

## Start here

Open [the notebook](notebooks/email_triage.ipynb), or inspect the
[executed snapshot and first live results](results/README.md). It walks through:

1. A reusable task signature and a live Databricks-backed `Predict` call.
2. A manual threshold that leaves the model probability unchanged.
3. Configurable costs: missed reply = 2, unnecessary alert = 1 by default.
4. Actual `ReAnchor.compile()` on training data, starting at threshold `0.5`.
5. A comparison on a separate, untouched test set.
6. Context-based selection between explicitly configured policies.
7. Saving and restoring the task and fitted policy without serializing credentials.

ReAnchor typically runs during development or periodic tuning. At runtime, the
application can use its fitted threshold or select among separately evaluated
policies based on supplied context. ReAnchor does not infer business priorities
or that context. This notebook's contextual thresholds are illustrative fixed
policies, not separately fitted policies or recommendations for real inboxes.

## Requirements and scope

- A Databricks workspace where `ai_decide` is available and enabled; it is **Beta**.
- Databricks serverless notebook compute, or local Python with Databricks CLI authentication.
- **DSPy 3.4.1** and **Databricks SDK 0.157.0**, pinned in `requirements.txt`.
- The decision types and ReAnchor interfaces are **experimental**.

The included `AiDecide` client subclasses the `TypeSafe` decision client present
in DSPy 3.4.1 and reuses its private cache/history helpers. It sends inference
requests only to Databricks. It is a small experimental integration, **not an
official connector**. Newer development documentation describes a different
interface; upgrading requires checking compatibility again.

Only synchronous binary `Noul` output is supported. Score, Choice, async,
streaming, prompt optimization, and a production connector are outside scope.
ReAnchor fits decision settings; it does not train the model or change its
probability estimates. A direct SQL/Python caller can also apply thresholds;
DSPy provides task organization, policy representation, and optimizer integration.

## Run in Databricks

1. Clone this repository into a **Git folder** after connecting your GitHub account
   under Settings → Linked accounts. Alternatively, upload the whole project.
2. Open `notebooks/email_triage.ipynb` and attach serverless compute.
3. Run all cells from the top. The first cells install pinned dependencies and
   restart Python. Authentication uses the notebook's workspace identity.
4. Read the test comparison and recorded checks at the bottom.

No tokens are embedded in the code. The notebook finds the repository root from
its location. If that fails, set its `project_root` widget to the workspace path
containing `ai_decide_client.py`. The two error-cost widgets are configurable.
The example does not read real email or create/change any database table.

The notebook writes result JSON to its temporary local directory. On serverless,
these files are session-local. The optional runner below collects them before
the session ends, including raw synthetic inference evidence and the saved policy.

### Optional: submit and collect from your terminal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
databricks auth login --host https://YOUR-WORKSPACE --profile YOUR_PROFILE
python tools/run_databricks.py submit --profile YOUR_PROFILE
# After the returned run has completed:
python tools/run_databricks.py collect --profile YOUR_PROFILE --run-id RUN_ID
```

This helper uploads the project into your user workspace folder and submits a
one-time serverless notebook run. It does not configure a Git provider connection
or create a persistent job. Run artifacts are collected under `results/runs/`
(ignored by Git). A running run returns its state; collect again after completion.

## Data and evaluation

The fixture contains **72 hand-authored synthetic emails**, fixed before inference:
36 training, 18 validation, and 18 test examples. See [the data notes](data/README.md).
Only sender, subject, body, and thread history enter model requests. Labels,
rationales, customer tier, and split membership do not.

The metric minimizes `2 × missed replies + 1 × unnecessary alerts` by default,
expressed as a normalized score to maximize. Validation is reported by ReAnchor;
the installed optimizer fits on training examples and uses its internal fold check.
The final test set is evaluated after all policies have been frozen.

Improvement is not guaranteed. The default threshold may already be suitable, or
a fitted setting may fail to improve on held-out emails. Report missed replies,
false alerts, and weighted error cost together. The small balanced synthetic
fixture is not representative of a real inbox and cannot establish real-world
accuracy, financial savings, or provider superiority.

Caching is enabled in memory and disk caching is disabled. Identical requests
reuse probabilities within a run. Recorded call counts are logical SDK calls,
not a billing measure; transport retries are not counted separately. Timing is
diagnostic, not a latency benchmark. Reruns can return different model probabilities.

## Local checks and notebook maintenance

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
# Regenerate the notebook only when changing its maintained source:
python tools/build_notebook.py
```

Offline tests use fake responses solely to check integration boundaries and
metric arithmetic. They are not evidence about Databricks model quality. The
live notebook is the end-to-end integration check. The dataset generator is
`tools/make_dataset.py`; do not change labels or split membership after inspecting
test outcomes and then claim the same test remains held out.

## References

- [Building & Optimizing Jev Programs with DSPy](https://www.cmpnd.ai/blog/building-jev-programs-with-dspy.html), Drew Breunig and Isaac Miller.
- [Databricks ai_decide](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_decide).
- [Databricks Git folders](https://learn.microsoft.com/en-us/azure/databricks/repos/git-operations-with-repos).
- [DSPy ReAnchor documentation](https://dspy.ai/current/api/experimental/ReAnchor/) — current docs may differ from this pinned release.

