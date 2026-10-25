"""Generate the single reader-facing notebook with nbformat."""

from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# DSPy Predict and ReAnchor with Databricks ai_decide

## Goal
Run an email decision task through `Predict`, apply explicit business policies,
and fit a threshold with the **actual, unmodified ReAnchor optimizer**.

The small custom client in `ai_decide_client.py` connects the two systems.
All model probabilities come from Databricks; no Jev or generative model is called.
`ai_decide` is **Beta**. DSPy's decision interfaces and ReAnchor are **experimental**.
This notebook deliberately pins DSPy 3.4.1, whose client interface differs from
the current development documentation.

### Assumptions
- All 72 emails are hand-authored synthetic examples, not a real inbox or benchmark.
- A missed reply costs 2 units and an unnecessary alert costs 1 by default.
- Train/validation/test membership was fixed before inference. Test labels are never used to fit policies.
- Only synchronous binary `Noul` decisions are supported by this example client.
""")
md("""
## Setup
Clone the repository into a Databricks Git folder, or upload the **whole project**.
Run on serverless notebook compute in a workspace where `ai_decide` is available
and enabled. Authentication uses the notebook's Databricks identity; do not paste tokens.
Locally, install `requirements.txt`, configure Databricks CLI authentication, and
set `DATABRICKS_CONFIG_PROFILE` to the intended workspace profile.
""")
code("""
%pip install dspy==3.4.1 databricks-sdk==0.157.0
""")
code("""
# Restart only inside Databricks so subsequent cells use the installed versions.
if "dbutils" in globals():
    dbutils.library.restartPython()
""")
code("""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dspy.experimental import ReAnchor
from databricks.sdk import WorkspaceClient
from IPython.display import display, HTML

if "dbutils" in globals():
    dbutils.widgets.text("project_root", "")
    dbutils.widgets.text("export_for_job", "false")
    dbutils.widgets.text("missed_reply_cost", "2")
    dbutils.widgets.text("false_alert_cost", "1")
    root_override = dbutils.widgets.get("project_root")
    missed_reply_cost = float(dbutils.widgets.get("missed_reply_cost"))
    false_alert_cost = float(dbutils.widgets.get("false_alert_cost"))
else:
    root_override = ""
    missed_reply_cost, false_alert_cost = 2.0, 1.0

search_from = Path(root_override).resolve() if root_override else Path.cwd().resolve()
project_root = next((p for p in [search_from, *search_from.parents]
                     if (p / "ai_decide_client.py").is_file()), None)
if project_root is None:
    raise FileNotFoundError("Clone/upload the full project, or set the project_root widget.")
sys.path.insert(0, str(project_root))

from ai_decide_client import AiDecide
from email_triage import (INPUTS, EmailTriage, predictor, business_metric, load_data,
                          examples, predict_rows, summarize, contextual_decisions)

assert metadata.version("dspy") == "3.4.1", "Use this example's pinned DSPy version."
assert metadata.version("databricks-sdk") == "0.157.0"
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
workspace = WorkspaceClient()
client = AiDecide(workspace)
baseline = predictor(client)  # The ordinary threshold is 0.5.
metric = business_metric(missed_reply_cost, false_alert_cost)
costs = {"missed_reply_cost": missed_reply_cost, "false_alert_cost": false_alert_cost}
splits = load_data(project_root / "data" / "synthetic_emails.jsonl")
dataset_hash = hashlib.sha256((project_root / "data" / "synthetic_emails.jsonl").read_bytes()).hexdigest()
source_hashes = {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
                 for name in ("ai_decide_client.py", "email_triage.py", "requirements.txt")}
optimizer_hash = hashlib.sha256(inspect.getsource(ReAnchor).encode()).hexdigest()
output_dir = Path(tempfile.mkdtemp(prefix="dspy-ai-decide-"))
print("Versions:", metadata.version("dspy"), metadata.version("databricks-sdk"))
print("Split sizes:", {k: len(v) for k, v in splits.items()})
print("Costs:", costs)
""")
md("""
## 1. Predict: inspect a probability and a decision
The signature specifies the task. Our client supplies Databricks probabilities.
The explicit `Noul` output exposes both the probability and the boolean obtained
by comparing it with the threshold. Labels, split membership, and customer tier
are excluded from model inputs.
""")
code("""
sample = splits["train"][0]
sample_inputs = {key: sample[key] for key in INPUTS}
answer = baseline(**sample_inputs).needs_response
raw_probability = client.call_log[-1]["raw_response"]["response"]["answers"]["needs_response"]["probability"]
assert answer.probability == raw_probability  # The client preserves the model evidence.
assert bool(answer) == (answer.probability >= 0.5)
print(json.dumps({"subject": sample["subject"], "probability": answer.probability,
                  "threshold": 0.5, "needs_response": bool(answer)}, indent=2))
""")
md("""
## 2. Set a policy manually
Lowering a threshold can catch more requests and can also create more false alerts.
The numbers below are illustrative policies, selected before seeing test results.
The probability does not change, and identical requests use the in-memory cache.
""")
code("""
manual = predictor(client, threshold=0.3)
calls_before_manual = len(client.call_log)
manual_answer = manual(**sample_inputs).needs_response
assert manual_answer.probability == answer.probability
assert len(client.call_log) == calls_before_manual
print(json.dumps({"probability": manual_answer.probability, "threshold": 0.3,
                  "needs_response": bool(manual_answer), "additional_backend_calls": 0}, indent=2))
""")
md("""
## 3. Express the business objective
Every missed reply costs `missed_reply_cost`; every unnecessary alert costs
`false_alert_cost`. Correct decisions cost zero. ReAnchor maximizes
`1 - error_cost / max(error_costs)` averaged across examples, which is equivalent
to minimizing average error cost. The chosen 2:1 ratio is an example assumption,
not a measured business value or a requirement to use that ratio.
""")
code("""
train_examples = examples(splits["train"])
validation_examples = examples(splits["validation"])
baseline_train = predict_rows(baseline, splits["train"])
baseline_validation = predict_rows(baseline, splits["validation"])
print("Training baseline:", summarize(baseline_train, **costs))
print("Validation baseline:", summarize(baseline_validation, **costs))
""")
md("""
## 4. Fit the threshold with actual ReAnchor
ReAnchor runs during development or periodic tuning. It fits the numeric policy
using labeled training examples and your metric; it does not rewrite the task,
train the model, or alter its probabilities. Validation examples are scored for
the report; the installed optimizer does not fit on them. Its internal fold check
can reject a candidate and retain the original threshold. Improvement is not required
for this example to demonstrate a working integration.
""")
code("""
calls_before_compile = len(client.call_log)
started = time.perf_counter()
optimizer = ReAnchor(metric=metric, num_threads=1)
optimized = optimizer.compile(baseline, trainset=train_examples, valset=validation_examples)
compile_seconds = time.perf_counter() - started
compile_additional_calls = len(client.call_log) - calls_before_compile
fitted_threshold = optimized.fields.get("needs_response", {}).get("threshold", 0.5)
assert getattr(optimized, "_compiled", False)
assert optimizer.report["fitted"]
assert hashlib.sha256(inspect.getsource(ReAnchor).encode()).hexdigest() == optimizer_hash
assert baseline.fields["needs_response"]["threshold"] == 0.5
print("Fitted threshold:", fitted_threshold)
print("Additional calls during compilation:", compile_additional_calls)
print(json.dumps(optimizer.report, indent=2, default=str))
""")
md("""
## 5. Freeze policies, then evaluate the final test set
The policies are fixed before test inference: baseline `0.5`, manual `0.3`,
ReAnchor's fitted threshold, and an illustrative contextual policy (`0.3` for
critical customers, `0.5` otherwise). Customer tiers are synthetic metadata,
not inferred by DSPy. Contextual thresholds are explicitly chosen rather than
separately optimized. Report their results without selecting a winner using the test set.
""")
code("""
frozen_policies = {"baseline": 0.5, "manual": 0.3, "reanchor": fitted_threshold,
                   "contextual": {"critical": 0.3, "standard": 0.5}}
calls_before_test = len(client.call_log)
baseline_test = predict_rows(baseline, splits["test"])
optimized_test = predict_rows(optimized, splits["test"])
manual_test = predict_rows(manual, splits["test"])
contextual_test = contextual_decisions(baseline_test, critical_threshold=0.3, standard_threshold=0.5)
assert len(client.call_log) - calls_before_test == len(splits["test"])
for original, fitted in zip(baseline_test, optimized_test):
    assert original["id"] == fitted["id"]
    assert original["probability"] == fitted["probability"]
    assert fitted["decision"] == (fitted["probability"] >= fitted_threshold)

test_predictions = {"baseline": baseline_test, "manual": manual_test,
                    "reanchor": optimized_test, "contextual": contextual_test}
test_metrics = {name: summarize(rows, **costs) for name, rows in test_predictions.items()}
# Exact counts make the tradeoff visible; this is a small synthetic example.
table_rows = "".join(
    f"<tr><td>{name}</td><td>{m['missed_replies']}</td><td>{m['false_alerts']}</td>"
    f"<td>{m['total_error_cost']:g}</td><td>{m['business_score']:.3f}</td></tr>"
    for name, m in test_metrics.items()
)
display(HTML("<h3>Held-out synthetic test emails</h3><table><tr><th>Policy</th>"
             "<th>Missed replies</th><th>False alerts</th><th>Error cost</th>"
             "<th>Business score</th></tr>" + table_rows + "</table>"))
print("Frozen policies:", frozen_policies)
""")
md("""
## 6. Verify dynamic policy selection through Predict
For each request, the application selects a configured predictor from supplied
business context. Keep a predictor for each policy instead of mutating a shared
predictor between requests. This check uses the same cached evidence and compares Predict's
decisions with the explicit `probability >= threshold` calculation.

Separately fitted policies could be selected in the same way once evaluated.
ReAnchor does not automatically infer context or business priorities.
""")
code("""
calls_before_context = len(client.call_log)
policy_programs = {tier: predictor(client, threshold=threshold)
                   for tier, threshold in frozen_policies["contextual"].items()}
for email, expected in zip(splits["test"], contextual_test):
    selected_program = policy_programs[email["customer_tier"]]
    result = selected_program(**{key: email[key] for key in INPUTS}).needs_response
    assert result.probability == expected["probability"]
    assert bool(result) == expected["decision"]
assert len(client.call_log) == calls_before_context
assert baseline.fields["needs_response"]["threshold"] == 0.5
print("Contextual policy selection matched the explicit rule for every test email.")
""")
md("""
## 7. Save only the fitted task and policy, then restore
Detach the authenticated client before saving. Reattach it at runtime after
loading, and verify that the restored program produces the same results.
The saved JSON contains no credentials or runtime client connection.
""")
code("""
saved_policy_path = output_dir / "optimized_policy.json"
optimized.set_lm(None)
try:
    optimized.save(str(saved_policy_path))
finally:
    optimized.set_lm(client)
restored = dspy.Predict(EmailTriage)
restored.load(str(saved_policy_path))
restored.set_lm(client)
assert predict_rows(restored, splits["validation"]) == predict_rows(optimized, splits["validation"])
print("Saved and restored policy:", restored.fields)
""")
md("""
## Checks and recorded results
The report preserves split sizes, costs, versions, source hashes, original
probabilities, fitted settings, optimizer evidence, and logical backend-call counts.
SDK retries are not separate entries in that counter. Timing includes orchestration
and is not a latency or cost benchmark. Local result files are temporary on serverless;
the job runner retrieves the report and evidence before the session ends.
""")
code("""
baseline_cost = test_metrics["baseline"]["total_error_cost"]
fitted_cost = test_metrics["reanchor"]["total_error_cost"]
outcome = "lower" if fitted_cost < baseline_cost else "higher" if fitted_cost > baseline_cost else "unchanged"
report = {
    "status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(), "dspy_version": metadata.version("dspy"),
    "databricks_sdk_version": metadata.version("databricks-sdk"), "ai_decide_version": "1.0",
    "data": {"synthetic": True, "sha256": dataset_hash,
             "split_sizes": {k: len(v) for k, v in splits.items()}},
    "source_sha256": source_hashes, "reanchor_source_sha256": optimizer_hash,
    "costs": costs, "policies": frozen_policies,
    "optimizer_report": optimizer.report,
    "test_metrics": test_metrics,
    "calls": {"total": len(client.call_log), "additional_during_compile": compile_additional_calls,
              "note": "Logical calls reaching the SDK; SDK transport retries are not counted separately."},
    "compile_seconds": compile_seconds,
    "checks": {"raw_probability_preserved": True, "manual_cache_reused": True,
               "test_probabilities_unchanged": True, "contextual_predict_matches_rule": True,
               "save_load": True, "actual_unmodified_reanchor": True},
    "finding": f"ReAnchor test error cost was {outcome} versus threshold 0.5 on these synthetic emails.",
    "limitations": ["Synthetic examples with labels authored by the example's developer, not a benchmark",
                    "Not a native official connector; pinned experimental DSPy interface",
                    "Only synchronous binary Noul; no Score, Choice, async, or provider comparison",
                    "No model training, changed probabilities, or real-world benefit claim"],
}
artifacts = {
    "evaluation_summary.json": report,
    "predictions.json": {"train_baseline": baseline_train, "validation_baseline": baseline_validation,
                         "test": test_predictions},
    "live_call_evidence.json": client.call_log,
    "optimized_policy.json": json.loads(saved_policy_path.read_text()),
}
for filename, value in artifacts.items():
    (output_dir / filename).write_text(json.dumps(value, indent=2, default=str))
print(report["finding"])
print("Logical live calls:", len(client.call_log), "| Compilation additional calls:", compile_additional_calls)
print("All checks passed. Result files:", output_dir)
if "dbutils" in globals() and dbutils.widgets.get("export_for_job").lower() == "true":
    dbutils.notebook.exit(json.dumps(artifacts, default=str))
""")
md("""
## Next steps
Read the recorded errors and tradeoffs before using a threshold in an application.
For a real workload, replace the synthetic fixture with appropriately labeled,
representative data, choose business costs explicitly, and evaluate the final
policy on unseen examples. Changing the model, task, or business context can
require reevaluation and periodic fitting. Keep real inbox data out of this repository.

References: [ai_decide](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_decide),
[original Jev/DSPy article](https://www.cmpnd.ai/blog/building-jev-programs-with-dspy.html).
For release-specific behavior, inspect the pinned DSPy 3.4.1 installation;
the current development docs describe a newer client interface.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
# Stable cell IDs keep regenerated source diffs readable.
for index, cell in enumerate(notebook.cells):
    cell.id = f"email-triage-{index:02d}"
nbf.validate(notebook)
target = root / "notebooks" / "email_triage.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(f"Wrote {target} ({len(cells)} cells)")
