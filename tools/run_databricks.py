"""Upload the example, submit one serverless run, or collect a completed run.

Use when a private Git folder connection is not configured yet. No persistent
job or cluster is created. Existing project files at the target path are updated.
"""
import argparse
import hashlib
import json
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit", "collect"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace-root")
    parser.add_argument("--run-id", type=int)
    args = parser.parse_args()
    w = WorkspaceClient(profile=args.profile)
    runs = ROOT / "results" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if args.action == "submit":
        workspace_root = args.workspace_root or f"/Workspace/Users/{w.current_user.me().user_name}/dspy-ai-decide-examples"
        for relative in ("ai_decide_client.py", "email_triage.py", "requirements.txt",
                         "data/synthetic_emails.jsonl", "notebooks/email_triage.ipynb"):
            target = f"{workspace_root}/{relative}"
            w.workspace.mkdirs(str(Path(target).parent))
            if relative.endswith(".ipynb"):
                target = target.removesuffix(".ipynb")
                format_ = ImportFormat.JUPYTER
            else:
                format_ = ImportFormat.RAW
            w.workspace.upload(target, (ROOT / relative).read_bytes(), format=format_, overwrite=True)
        submission = w.jobs.submit(
            run_name="DSPy Predict and ReAnchor email example",
            timeout_seconds=1200,
            tasks=[jobs.SubmitTask(
                task_key="email_triage", timeout_seconds=1200, max_retries=0,
                notebook_task=jobs.NotebookTask(
                    notebook_path=f"{workspace_root}/notebooks/email_triage", source=jobs.Source.WORKSPACE,
                    base_parameters={"project_root": workspace_root, "export_for_job": "true"},
                ),
            )],
        )
        run_id = submission.response.run_id
        record = {"run_id": run_id, "workspace_root": workspace_root, "workspace_host": w.config.host,
                  "notebook_sha256": hashlib.sha256((ROOT / "notebooks/email_triage.ipynb").read_bytes()).hexdigest()}
        (runs / f"submission_{run_id}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        return
    if not args.run_id:
        parser.error("collect requires --run-id")
    run = w.jobs.get_run(args.run_id)
    state = run.state.as_dict() if run.state else {}
    print(json.dumps({"run_id": args.run_id, "state": state, "run_page_url": run.run_page_url}, indent=2))
    if not run.state or run.state.life_cycle_state != jobs.RunLifeCycleState.TERMINATED:
        return
    task = run.tasks[0]
    output = w.jobs.get_run_output(task.run_id)
    destination = runs / str(args.run_id)
    destination.mkdir(exist_ok=True)
    (destination / "run_metadata.json").write_text(json.dumps(run.as_dict(), indent=2))
    (destination / "run_output.json").write_text(json.dumps(output.as_dict(), indent=2))
    if run.state.result_state != jobs.RunResultState.SUCCESS:
        raise RuntimeError(output.error or state)
    if not output.notebook_output or output.notebook_output.truncated:
        raise RuntimeError("Notebook output missing or truncated; cannot verify the evidence.")
    artifacts = json.loads(output.notebook_output.result)
    expected = {"evaluation_summary.json", "predictions.json", "live_call_evidence.json", "optimized_policy.json"}
    if set(artifacts) != expected or artifacts["evaluation_summary.json"]["status"] != "PASS":
        raise RuntimeError("Unexpected notebook result.")
    for filename, contents in artifacts.items():
        (destination / filename).write_text(json.dumps(contents, indent=2))
    print(json.dumps(artifacts["evaluation_summary.json"], indent=2))
    print("Collected verified artifacts:", destination)


if __name__ == "__main__":
    main()
