"""Minimal synchronous binary client for the blog, pinned to DSPy 3.4.1.

Reuses the experimental TypeSafe decision contract and cache helpers, but all
inference goes to Databricks. This is not an official or general-purpose connector.
"""

import math
import time

import dspy
from dspy.experimental import TypeSafe


class AiDecide(TypeSafe):
    def __init__(self, workspace, *, cache=True):
        super().__init__(model="ai_decide", base_url=workspace.config.host, cache=cache)
        self.workspace = workspace
        self.call_log = []

    def __deepcopy__(self, memo):
        # ReAnchor copies the predictor; share its authenticated runtime connection
        # and call recorder. Neither is part of the learned decision policy.
        memo[id(self)] = self
        return self

    def __call__(self, state, questions):
        if not questions or any(q["type"] != "noul" for q in questions.values()):
            raise ValueError("This example client supports binary Noul questions only.")
        request = {
            **self._request(state, questions),
            "provider": "databricks-ai-decide",
            "version": "1.0",
        }
        response = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if not cache_hit:
            started = time.perf_counter()
            raw = self.workspace.api_client.do(
                method="POST",
                path="/api/2.0/ai-functions/ai-decide",
                body={"state": state, "questions": questions, "options": {"version": "1.0"}},
            )
            if raw.get("error_message"):
                raise RuntimeError(raw["error_message"])
            answers = {}
            for name in questions:
                answer = raw["response"]["answers"][name]
                p = answer["probability"]
                if (answer["type"] != "noul" or isinstance(p, bool)
                        or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1):
                    raise ValueError(f"Invalid Noul probability for {name!r}.")
                answers[name] = {"noul": p}
            self.call_log.append({
                "state": state, "questions": questions, "raw_response": raw,
                "elapsed_seconds": time.perf_counter() - started,
            })
            response = {"model": self.model, "usage": {}, "answers": answers}
        return self._finish(request, response, cache_hit)

    async def acall(self, state, questions):
        raise NotImplementedError("This blog client is synchronous; async is out of scope.")


