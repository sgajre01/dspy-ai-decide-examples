"""Focused offline checks for the integration contract and business calculations.

The fake service below validates plumbing only; it is not model evidence.
The live Databricks notebook provides the separate integration check.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dspy
from ai_decide_client import AiDecide
from email_triage import (INPUTS, business_metric, contextual_decisions, examples,
                          load_data, predictor, summarize)


class FakeAPI:
    def __init__(self, probability):
        self.probability, self.requests = probability, []

    def do(self, **request):
        self.requests.append(request)
        return {"response": {"answers": {"needs_response": {
            "type": "noul", "probability": self.probability}}}, "metadata": {"version": "1.0"}}


class DecisionChecks(unittest.TestCase):
    def setUp(self):
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)

    def test_probability_is_preserved_and_threshold_is_inclusive(self):
        api = FakeAPI(0.4)
        client = AiDecide(SimpleNamespace(config=SimpleNamespace(host="https://unit-test.invalid"), api_client=api))
        inputs = dict(sender="Test", subject="Test", body="Please answer", history="Open request")
        low = predictor(client, 0.4)(**inputs).needs_response
        high = predictor(client, 0.5)(**inputs).needs_response
        self.assertEqual((low.probability, high.probability), (0.4, 0.4))
        self.assertTrue(bool(low))
        self.assertFalse(bool(high))
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.requests[0]["path"], "/api/2.0/ai-functions/ai-decide")
        self.assertNotIn("threshold", str(api.requests[0]["body"]))

    def test_zero_probability_is_valid(self):
        api = FakeAPI(0.0)
        client = AiDecide(SimpleNamespace(config=SimpleNamespace(host="https://zero.invalid"), api_client=api))
        result = predictor(client, 0.0)(sender="A", subject="B", body="C", history="D")
        self.assertEqual(result.needs_response.probability, 0.0)
        self.assertTrue(bool(result.needs_response))

    def test_invalid_probability_fails(self):
        for p in (True, -0.1, 1.1, float("nan")):
            client = AiDecide(SimpleNamespace(config=SimpleNamespace(host="https://invalid.invalid"), api_client=FakeAPI(p)), cache=False)
            with self.assertRaises(ValueError):
                client(state={}, questions={"needs_response": {"type": "noul"}})

    def test_business_costs_reconcile(self):
        rows = [{"expected": y, "decision": p} for y, p in [(True, True), (True, False), (False, True), (False, False)]]
        result = summarize(rows)
        self.assertEqual(result["missed_replies"], 1)
        self.assertEqual(result["false_alerts"], 1)
        self.assertEqual(result["total_error_cost"], 3)
        self.assertEqual(result["business_score"], 0.625)
        metric = business_metric()
        observed = sum(metric(SimpleNamespace(needs_response=r["expected"]),
                              SimpleNamespace(needs_response=r["decision"])) for r in rows) / len(rows)
        self.assertEqual(observed, result["business_score"])

    def test_context_changes_action_without_changing_probability(self):
        rows = [{"probability": 0.4, "customer_tier": tier} for tier in ("critical", "standard")]
        result = contextual_decisions(rows)
        self.assertEqual([r["decision"] for r in result], [True, False])
        self.assertEqual([r["probability"] for r in result], [0.4, 0.4])

    def test_fixture_splits_and_no_label_leakage(self):
        splits = load_data(Path(__file__).resolve().parents[1] / "data/synthetic_emails.jsonl")
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 36, "validation": 18, "test": 18})
        for rows in splits.values():
            for example in examples(rows):
                self.assertEqual(set(example.inputs().keys()), set(INPUTS))


if __name__ == "__main__":
    unittest.main()
