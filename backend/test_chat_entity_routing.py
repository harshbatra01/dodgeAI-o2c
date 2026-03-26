import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


def _read_sse_payload(response_text: str) -> tuple[str, list[str]]:
    token = ""
    referenced_nodes: list[str] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "token" in data:
            token = data["token"]
        if "referenced_nodes" in data:
            referenced_nodes = data["referenced_nodes"]
    return token, referenced_nodes


class ChatEntityRoutingTests(unittest.TestCase):
    def test_entity_routing_regressions(self) -> None:
        cases = [
            {
                "query": "tell me about any ranodm node",
                "assertion": lambda token, refs: self.assertTrue(bool(refs), "Expected at least one referenced node"),
            },
            {
                "query": "tell me about any rnadom plant",
                "assertion": lambda token, refs: self.assertTrue(
                    refs and refs[0].startswith("Plant:"), f"Expected Plant node, got {refs}"
                ),
            },
            {
                "query": "tell me about any customer",
                "assertion": lambda token, refs: self.assertTrue(
                    refs and refs[0].startswith("Customer:"), f"Expected Customer node, got {refs}"
                ),
            },
            {
                "query": "tell me about any random customer",
                "assertion": lambda token, refs: self.assertTrue(
                    refs and refs[0].startswith("Customer:"), f"Expected Customer node, got {refs}"
                ),
            },
            {
                "query": "tell me about customer jordan",
                "assertion": lambda token, refs: (
                    self.assertIn("**Customer**", token),
                    self.assertTrue(any(node_id.startswith("Customer:") for node_id in refs)),
                ),
            },
            {
                "query": "BillingDocument:90504272",
                "assertion": lambda token, refs: self.assertIn("BillingDocument:90504272", token),
            },
            {
                "query": "tell me about customer xyz123",
                "assertion": lambda token, refs: self.assertEqual(token, "No matching entity found in dataset"),
            },
            {
                "query": "How do I bake a chocolate cake?",
                "assertion": lambda token, refs: self.assertIn(
                    "designed to answer dataset-related queries only", token
                ),
            },
        ]

        with TestClient(app) as client:
            memory: list[dict[str, list[str] | str]] = []
            for case in cases:
                payload = {"messages": [{"role": "user", "content": case["query"]}]}
                if memory:
                    payload["memory"] = {"turns": memory}
                response = client.post("/chat", json=payload)
                self.assertEqual(response.status_code, 200)
                token, refs = _read_sse_payload(response.text)
                self.assertIsNotNone(token)
                case["assertion"](token, refs)
                memory.append({
                    "referenced_nodes": refs,
                    "user_message": case["query"],
                    "assistant_message": token,
                })

    def test_pronoun_followups(self) -> None:
        with TestClient(app) as client:
            first = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "tell about Customer:310000109"}]},
            )
            first_token, first_refs = _read_sse_payload(first.text)
            self.assertIn("Customer:310000109", first_token)
            self.assertEqual(first_refs, ["Customer:310000109"])

            memory = {
                "turns": [
                    {
                        "referenced_nodes": first_refs,
                        "user_message": "tell about Customer:310000109",
                        "assistant_message": first_token,
                    }
                ]
            }

            second = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "tell about its relationships"}], "memory": memory},
            )
            second_token, second_refs = _read_sse_payload(second.text)
            self.assertIn("Connected entities", second_token)
            self.assertTrue(second_refs)

            third = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "tell more details about it"}], "memory": memory},
            )
            third_token, third_refs = _read_sse_payload(third.text)
            self.assertIn("**", third_token)
            self.assertTrue(third_refs)

    def test_llm_fallback_for_complex_query(self) -> None:
        """Complex queries should reach the LLM, not be swallowed by deterministic routing."""
        with TestClient(app) as client, patch(
            "routes.chat._call_llm_query_planner",
            return_value={
                "intent": "aggregate",
                "steps": [
                    {
                        "action": "aggregate",
                        "params": {"node_type": "Customer", "group_by": "name", "top_n": 5},
                    }
                ],
            },
        ) as mock_llm:
            response = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "which clients have the most invoice documents?"}]},
            )
            token, refs = _read_sse_payload(response.text)
            # The LLM should have been called since this is a complex query
            self.assertTrue(mock_llm.called, "LLM should be called for complex queries")
            self.assertIn("Top results", token)
            self.assertTrue(any(node_id.startswith("Customer:") for node_id in refs))

    def test_llm_off_topic_does_not_override_dataset_like_query(self) -> None:
        with TestClient(app) as client, patch(
            "routes.chat._call_llm_query_planner",
            return_value={"intent": "off_topic", "message": "This should not be used"},
        ):
            response = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "which clients have the most invoice documents?"}]},
            )
            token, _ = _read_sse_payload(response.text)
            # Should NOT say off_topic for a dataset-related query
            self.assertNotEqual(token, "This should not be used")

    def test_llm_quota_error_message(self) -> None:
        with TestClient(app) as client, patch(
            "routes.chat._call_llm_query_planner",
            return_value={"intent": "service_unavailable", "message": "Service temporarily unavailable due to usage limits"},
        ):
            response = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "which clients have the most invoice documents?"}]},
            )
            token, _ = _read_sse_payload(response.text)
            self.assertEqual(token, "Service temporarily unavailable due to usage limits")

    def test_complex_query_reaches_llm_not_entity_lookup(self) -> None:
        """When user asks 'what is the most billed order of this customer', the LLM should be called."""
        with TestClient(app) as client, patch(
            "routes.chat._call_llm_query_planner",
            return_value={
                "intent": "aggregate",
                "steps": [
                    {
                        "action": "customer_top_billed_orders",
                        "params": {"customer_id": "Customer:310000109", "top_n": 5},
                    }
                ],
            },
        ) as mock_llm:
            memory = {
                "turns": [
                    {
                        "referenced_nodes": ["Customer:310000109"],
                        "user_message": "tell about jordan",
                        "assistant_message": "**Customer** Customer:310000109",
                    }
                ]
            }
            response = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "what is the most billed order of this customer"}],
                    "memory": memory,
                },
            )
            token, refs = _read_sse_payload(response.text)
            self.assertTrue(mock_llm.called, "LLM should be called for 'most billed order' query")


if __name__ == "__main__":
    unittest.main()
