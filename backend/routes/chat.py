"""
chat.py — LLM-Powered Natural Language Interface (Phase 3)

Translates natural language Order-to-Cash queries into structured JSON
query plans via Groq (llama-3.3-70b-versatile), executes them against
the NetworkX graph, then asks the LLM to summarise the results.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import ssl
import urllib.error
import urllib.request
from difflib import SequenceMatcher, get_close_matches
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from dotenv import load_dotenv
import certifi
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from graph_query_engine import QueryEngine

logger = logging.getLogger(__name__)

# Load env vars (for GROQ_API_KEY)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

router = APIRouter(prefix="/chat", tags=["Query Agent"])
NODE_ID_PATTERN = re.compile(r"\b(Customer|Product|Plant|SalesOrder|SalesOrderItem|DeliveryDocument|DeliveryItem|BillingDocument|BillingItem|JournalEntry|Payment):[A-Za-z0-9_\-:]+\b")
NODE_TYPE_BY_PHRASE: dict[str, str] = {
    "customer": "Customer",
    "plant": "Plant",
    "product": "Product",
    "sales order": "SalesOrder",
    "billing document": "BillingDocument",
    "billing item": "BillingItem",
    "delivery document": "DeliveryDocument",
    "journal entry": "JournalEntry",
    "payment": "Payment",
}

OFF_TOPIC_HINTS = {
    "capital of",
    "weather",
    "poem",
    "joke",
    "story",
    "recipe",
    "president",
    "prime minister",
    "stock price",
    "football",
    "cricket",
}
ENTITY_QUALIFIER_TOKENS = {"random", "any", "some"}
ENTITY_ARTICLES = {"a", "an", "the"}
KEYWORD_LEXICON = {
    "random",
    "any",
    "some",
    "show",
    "list",
    "give",
    "get",
    "find",
    "what",
    "which",
    "has",
    "have",
    "had",
    "bought",
    "ordered",
    "purchased",
    "linked",
    "involved",
    "involvement",
    "activity",
    "activities",
    "revenue",
    "revenues",
    "invoice",
    "invoices",
    "billed",
    "bill",
    "bills",
    "most",
    "top",
    "highest",
    "largest",
    "biggest",
    "lifecycle",
    "trace",
    "flow",
    "flows",
    "details",
    "detail",
    "node",
    "entity",
    "customer",
    "plant",
    "product",
    "sales",
    "order",
    "billing",
    "document",
    "billingitem",
    "delivery",
    "journal",
    "entry",
    "payment",
    "relationship",
    "relationships",
    "related",
    "connection",
    "connections",
    "connected",
    "link",
    "links",
    "neighbor",
    "neighbors",
    "neighbour",
    "neighbours",
    "orders",
    "order",
}
RELATIONSHIP_TOKENS = {
    "relationship",
    "relationships",
    "related",
    "connection",
    "connections",
    "connected",
    "link",
    "links",
    "neighbor",
    "neighbors",
    "neighbour",
    "neighbours",
    "connections",
    "connection",
    "linked",
    "link",
    "links",
    "involved",
    "involvement",
    "activity",
    "activities",
}
ENTITY_TYPE_ALIASES: dict[str, str] = {
    "customer": "Customer",
    "customers": "Customer",
    "plant": "Plant",
    "plants": "Plant",
    "product": "Product",
    "products": "Product",
    "sales order": "SalesOrder",
    "sales orders": "SalesOrder",
    "order": "SalesOrder",
    "orders": "SalesOrder",
    "billing document": "BillingDocument",
    "billing documents": "BillingDocument",
    "billing": "BillingDocument",
    "invoice": "BillingDocument",
    "invoices": "BillingDocument",
    "billing item": "BillingItem",
    "billing items": "BillingItem",
    "delivery document": "DeliveryDocument",
    "delivery documents": "DeliveryDocument",
    "delivery": "DeliveryDocument",
    "journal entry": "JournalEntry",
    "journal entries": "JournalEntry",
    "payment": "Payment",
    "payments": "Payment",
}

INTENT_VERB_TOKENS = {
    "show",
    "list",
    "give",
    "get",
    "find",
    "what",
    "which",
    "who",
    "tell",
    "display",
}

ORDER_REQUEST_TOKENS = {
    "order",
    "orders",
    "bought",
    "buy",
    "buys",
    "ordered",
    "purchase",
    "purchased",
    "sold",
    "linked",
    "link",
    "related",
    "connections",
    "connection",
}

FLOW_REQUEST_TOKENS = {
    "flow",
    "flows",
    "lifecycle",
    "trace",
    "path",
    "journey",
    "process",
    "move",
    "through",
    "system",
}

CONDITIONAL_REQUEST_TOKENS = {
    "without",
    "missing",
    "incomplete",
    "broken",
    "not",
    "delivered",
    "billed",
    "payment",
    "payments",
    "invoice",
    "invoices",
}

RELATIONSHIP_REQUEST_TOKENS = RELATIONSHIP_TOKENS | {
    "connection",
    "connections",
    "interact",
    "interacts",
    "interaction",
    "interactions",
    "activity",
    "activities",
    "involved",
    "linked",
    "link",
    "links",
}

QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "with",
    "about",
    "me",
    "please",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "this",
    "that",
    "its",
    "their",
}

ALLOWED_ACTIONS = {
    "get_node",
    "get_nodes_by_ids",
    "get_neighbours",
    "get_subgraph",
    "search_nodes",
    "count_by_type",
    "filter_nodes",
    "filter_node_ids",
    "aggregate",
    "aggregate_connected_sum",
    "aggregate_delivery_activity",
    "top_billed_orders",
    "top_products_by_billing_documents",
    "customer_top_billed_orders",
    "customer_top_products",
    "random_node",
    "trace_flow",
    "find_connected",
    "find_delivered_without_billing",
    "find_billed_without_delivery",
    "find_invoices_without_payments",
    "find_incomplete_flows",
    "stats",
}

VALID_NODE_TYPES = {
    "Customer",
    "Product",
    "Plant",
    "SalesOrder",
    "SalesOrderItem",
    "DeliveryDocument",
    "DeliveryItem",
    "BillingDocument",
    "BillingItem",
    "JournalEntry",
    "Payment",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryTurn(BaseModel):
    referenced_nodes: list[str] = Field(default_factory=list)
    user_message: str | None = None
    assistant_message: str | None = None


class ConversationMemory(BaseModel):
    turns: list[MemoryTurn] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(
        ..., example=[{"role": "user", "content": "Which products appear in the most billing documents?"}]
    )
    memory: ConversationMemory | None = None

class ChatResponse(BaseModel):
    reply: str
    query_plan: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# System prompt — full schema, business context, few-shot examples
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Dodge AI, an expert assistant for SAP Order-to-Cash (O2C) process analysis.
You have access to a NetworkX graph containing 1,338 nodes across 11 types and 5,132 edges across 13 types, representing an end-to-end O2C dataset.

## GRAPH SCHEMA

### Node Types and Their Key Properties:
1. **Customer** — customer_id, name, category, city, country, region, companyCode, paymentTerms, salesAreaAssignments[]
2. **Product** — product_id, description, productType, productGroup, grossWeight, netWeight, weightUnit, baseUnit, division
3. **Plant** — plant_id, plantName, salesOrganization, distributionChannel, division
4. **SalesOrder** — salesOrder_id, salesOrderType, salesOrganization, totalNetAmount, transactionCurrency, creationDate, overallDeliveryStatus, requestedDeliveryDate, paymentTerms
5. **SalesOrderItem** — salesOrder_id, salesOrderItem_id, material, requestedQuantity, netAmount, transactionCurrency, materialGroup, productionPlant
6. **DeliveryDocument** — deliveryDocument_id, creationDate, goodsMovementDate, goodsMovementStatus, pickingStatus, shippingPoint
7. **DeliveryItem** — deliveryDocument_id, deliveryDocumentItem_id, actualDeliveryQuantity, plant, storageLocation
8. **BillingDocument** — billingDocument_id, billingDocumentType, billingDocumentDate, totalNetAmount, transactionCurrency, isCancelled, companyCode, fiscalYear
9. **BillingItem** — billingDocument_id, billingDocumentItem_id, material, billingQuantity, netAmount
10. **JournalEntry** — accountingDocument_id, accountingDocumentItem_id, glAccount, amountInTransactionCurrency, postingDate, clearingDate
11. **Payment** — accountingDocument_id, accountingDocumentItem_id, amountInTransactionCurrency, postingDate, clearingDate, invoiceReference

### Edge Types (directed):
1. SOLD_TO: SalesOrder → Customer
2. HAS_ITEM: SalesOrder → SalesOrderItem, DeliveryDocument → DeliveryItem, BillingDocument → BillingItem
3. CONTAINS_PRODUCT: SalesOrderItem → Product
4. AVAILABLE_AT: Product → Plant
5. FULFILLS_ORDER: DeliveryDocument → SalesOrder
6. SHIPS_FROM: DeliveryItem → Plant
7. BILLS_DELIVERY: BillingDocument → DeliveryDocument
8. BILLED_TO: BillingDocument → Customer
9. BILLS_PRODUCT: BillingItem → Product
10. CANCELLED_BY: BillingDocument → BillingDocument (cancellation link)
11. RECORDS_BILLING: JournalEntry → BillingDocument
12. FOR_CUSTOMER: JournalEntry → Customer
13. CLEARS_JOURNAL: Payment → JournalEntry
14. PAID_BY: Payment → Customer

### Node ID Format:
All node IDs follow the pattern `NodeType:value` or `NodeType:value1:value2`.
Examples: `Customer:17100001`, `SalesOrder:740506`, `BillingDocument:90504248`, `Product:TG11`, `Plant:1710`

## O2C PROCESS FLOW
The standard Order-to-Cash flow is:
Customer → SalesOrder → SalesOrderItem → Product
SalesOrder → DeliveryDocument → DeliveryItem
DeliveryDocument → BillingDocument → BillingItem
BillingDocument → JournalEntry → Payment

## AVAILABLE ACTIONS
You must produce a JSON query plan with an "intent" and a "steps" array. Each step has an "action" and "params":

1. **get_node** — Get a single node's details
   params: {"node_id": "BillingDocument:90504248"}

2. **get_neighbours** — Get all neighbours of a node, optionally filtered
   params: {"node_id": "...", "edge_type": "HAS_ITEM" (optional), "direction": "in"|"out"|"both" (default: both)}

3. **get_subgraph** — Get the ego graph around a node
   params: {"center": "...", "radius": 2}

4. **search_nodes** — Text search across node IDs and properties
   params: {"query": "TG11", "node_type": "Product" (optional), "limit": 20}

5. **count_by_type** — Count nodes by type
   params: {"node_type": "BillingDocument" (optional, omit for all)}

6. **filter_nodes** — Filter nodes by type and attribute conditions
   params: {"node_type": "SalesOrder", "filters": {"overallDeliveryStatus": "C"}, "limit": 20}

7. **aggregate** — Group-by aggregation on node attributes
   params: {"node_type": "BillingItem", "group_by": "material", "metric": "count", "top_n": 10}
   Also supports counting edges: {"node_type": "Product", "group_by": "description", "via_edge_type": "BILLS_PRODUCT", "count_connected_node_type": "BillingItem", "top_n": 10}

8. **trace_flow** — BFS traversal to trace the full O2C flow of a document
   params: {"node_id": "BillingDocument:90504248", "max_depth": 6}

9. **find_connected** — Find all nodes of a specific type reachable from a given node
   params: {"node_id": "...", "target_node_type": "Customer", "max_depth": 3}

10. **find_billed_without_delivery** — Custom logic to find SalesOrders that are billed but have no delivery document.
    params: {}

11. **find_delivered_without_billing** — Custom logic to find SalesOrders that have been delivered but lack billing documents.
    params: {}

12. **stats** — Get overall graph statistics
    params: {}

## FEW-SHOT EXAMPLES

### Example 1: "Which products appear in the most billing documents?"
```json
{
  "intent": "aggregate",
  "steps": [
    {
      "action": "aggregate",
      "params": {
        "node_type": "Product",
        "group_by": "description",
        "via_edge_type": "BILLS_PRODUCT",
        "count_connected_node_type": "BillingItem",
        "top_n": 10
      }
    }
  ]
}
```

### Example 2: "Trace the full flow for billing document 90504248"
```json
{
  "intent": "trace_document",
  "steps": [
    {
      "action": "trace_flow",
      "params": {
        "node_id": "BillingDocument:90504248",
        "max_depth": 6
      }
    }
  ]
}
```

### Example 3: "Show me details of customer 17100001"
```json
{
  "intent": "detail",
  "steps": [
    {
      "action": "get_node",
      "params": {"node_id": "Customer:17100001"}
    },
    {
      "action": "get_neighbours",
      "params": {"node_id": "Customer:17100001"}
    }
  ]
}
```

### Example 4: "Identify sales orders that have been delivered but not billed"
```json
{
  "intent": "find_missing_relationship",
  "steps": [
    {
      "action": "find_delivered_without_billing",
      "params": {}
    }
  ]
}
```

### Example 5: "Identify sales orders that are billed but have no delivery document"
```json
{
  "intent": "find_missing_relationship",
  "steps": [
    {
      "action": "find_billed_without_delivery",
      "params": {}
    }
  ]
}
```

## HARD RULES
1. **ONLY answer questions about the SAP Order-to-Cash dataset.** If the user asks anything unrelated (general knowledge, weather, politics, coding, etc.), you MUST return:
   ```json
   {"intent": "off_topic", "message": "I can only answer questions about the SAP Order-to-Cash dataset. Please ask about orders, deliveries, billing, customers, products, or payments."}
   ```
2. **ALWAYS return valid JSON** — no markdown, no commentary, no wrapping. Your entire response must be a single JSON object.
3. **Use the exact node ID format**: `NodeType:value` (e.g., `BillingDocument:90504248`, not just `90504248`).
4. **Be precise**: choose the minimum steps needed. Don't over-fetch.
5. **For product-billing analysis**: Products connect to BillingItems via the BILLS_PRODUCT edge (directed BillingItem → Product). Use the aggregate action with via_edge_type.
6. **Conversation context resolution**: Before generating a query plan, check the CONVERSATION CONTEXT section appended below for any entities referenced in prior turns. If the user uses pronouns or shorthand like "this customer", "that order", "the same document", "it", "them", "the top one", "the first one" — resolve them to the actual node ID from the conversation context and use that node ID directly in the query plan steps. NEVER hallucinate or guess a node ID — always use the exact ID from conversation context.
"""

def _node_label(store, node_id: str) -> str:
    attrs = dict(store.G.nodes[node_id])
    return (
        attrs.get("name")
        or attrs.get("description")
        or attrs.get("plantName")
        or attrs.get("material")
        or attrs.get("customer_id")
        or attrs.get("product_id")
        or attrs.get("salesOrder_id")
        or attrs.get("billingDocument_id")
        or attrs.get("deliveryDocument_id")
        or attrs.get("accountingDocument_id")
        or node_id.split(":", 1)[1]
    )


def _collect_fallback_memory_turns(messages: list[dict[str, str]], store) -> list[list[str]]:
    """Recover per-turn node references from assistant text when structured memory is absent."""
    turns: list[list[str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        seen = set()
        turn_nodes: list[str] = []
        for match in NODE_ID_PATTERN.finditer(msg.get("content", "")):
            node_id = match.group(0)
            if node_id in seen or not store.has_node(node_id):
                continue
            seen.add(node_id)
            turn_nodes.append(node_id)
        turns.append(turn_nodes)
    return turns


def _build_conversation_context(body: ChatRequest, store) -> dict[str, Any]:
    """Build structured conversation memory from explicit memory payload or assistant text."""
    turn_node_lists = [turn.referenced_nodes for turn in body.memory.turns] if body.memory else []
    if not turn_node_lists:
        turn_node_lists = _collect_fallback_memory_turns(body.messages, store)

    turns: list[dict[str, Any]] = []
    ordered_entities: list[dict[str, Any]] = []
    latest_by_type: dict[str, str] = {}

    for turn_index, node_ids in enumerate(turn_node_lists):
        filtered_nodes = []
        for rank, node_id in enumerate(node_ids):
            if not store.has_node(node_id):
                continue
            node_type = store.G.nodes[node_id].get("node_type", node_id.split(":", 1)[0])
            latest_by_type[node_type] = node_id
            ordered_entities.append({
                "node_id": node_id,
                "node_type": node_type,
                "label": _node_label(store, node_id),
                "turn_index": turn_index,
                "rank": rank,
            })
            filtered_nodes.append(node_id)
        turns.append({"node_ids": filtered_nodes})

    last_turn_nodes = turns[-1]["node_ids"] if turns else []
    return {
        "turns": turns,
        "ordered_entities": ordered_entities,
        "last_turn_nodes": last_turn_nodes,
        "latest_by_type": latest_by_type,
    }


def _find_matching_nodes(store, query: str, node_type: str | None = None, limit: int = 10) -> list[str]:
    def score_nodes(search_query: str) -> list[tuple[int, str]]:
        query_lower = search_query.lower().strip()
        query_parts = [part for part in re.split(r"\s+", query_lower) if part]
        scored_local: list[tuple[int, str]] = []

        for node_id, attrs in store.G.nodes(data=True):
            if node_type and attrs.get("node_type") != node_type:
                continue
            label = _node_label(store, node_id).lower()
            haystack = " ".join(
                [
                    node_id,
                    label,
                    *(str(value) for value in attrs.values() if isinstance(value, (str, int, float))),
                ]
            ).lower()
            if query_lower and query_lower in haystack:
                score = 100
                if query_lower == str(attrs.get("name", "")).lower():
                    score = 200
                elif query_lower == str(attrs.get("description", "")).lower():
                    score = 180
                scored_local.append((score, node_id))
                continue
            if query_parts and all(part in haystack for part in query_parts):
                score = 50 + len(query_parts)
                scored_local.append((score, node_id))
                continue

            fuzzy_source = label or haystack
            if len(query_lower) >= 3:
                ratio = SequenceMatcher(None, query_lower, fuzzy_source).ratio()
                if ratio >= 0.66:
                    score = int(40 + ratio * 40)
                    scored_local.append((score, node_id))
                    continue

            if len(query_parts) == 1 and len(query_lower) >= 3:
                token_best = 0.0
                for token in re.split(r"[\s,:]+", fuzzy_source):
                    if not token:
                        continue
                    token_best = max(token_best, SequenceMatcher(None, query_lower, token).ratio())
                if token_best >= 0.72:
                    score = int(35 + token_best * 45)
                    scored_local.append((score, node_id))
        return scored_local

    query_lower = query.lower().strip()
    scored = score_nodes(query_lower)

    if not scored:
        filtered_tokens = store.filter_query_tokens(query_lower, node_type=node_type)
        filtered_query = " ".join(filtered_tokens).strip()
        if filtered_query and filtered_query != query_lower:
            scored = score_nodes(filtered_query)

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _, node_id in scored[:limit]]


def _latest_entity_of_type(conv_context: dict[str, Any], node_type: str) -> str | None:
    if node_type in conv_context["latest_by_type"]:
        return conv_context["latest_by_type"][node_type]
    for entity in reversed(conv_context["ordered_entities"]):
        if entity["node_type"] == node_type:
            return entity["node_id"]
    return None


def _last_turn_entities_of_type(conv_context: dict[str, Any], store, node_type: str) -> list[str]:
    return [
        node_id for node_id in conv_context["last_turn_nodes"]
        if store.G.nodes[node_id].get("node_type", node_id.split(":", 1)[0]) == node_type
    ]


def _resolve_expected_node_type(user_message: str) -> str | None:
    text = user_message.lower()
    if "customer" in text or "client" in text or "clients" in text:
        return "Customer"
    if "billing document" in text or "invoice" in text:
        return "BillingDocument"
    if "order" in text:
        return "SalesOrder"
    if "product" in text:
        return "Product"
    if "payment" in text:
        return "Payment"
    if "journal" in text or "clearing document" in text:
        return "JournalEntry"
    if "delivery" in text:
        return "DeliveryDocument"
    return None


def _extract_direct_node_id(text: str) -> str | None:
    text = text.strip()
    if NODE_ID_PATTERN.fullmatch(text):
        return text
    match = NODE_ID_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def _resolve_node_type_phrase(text: str) -> str | None:
    lowered = text.lower()
    for phrase in sorted(ENTITY_TYPE_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return ENTITY_TYPE_ALIASES[phrase]
    for phrase in sorted(NODE_TYPE_BY_PHRASE.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return NODE_TYPE_BY_PHRASE[phrase]
    return None


def _query_implies_flow(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "lifecycle",
            "move through the system",
            "move through",
            "flow of",
            "full flow",
            "trace the flow",
            "how does",
            "process",
            "path",
            "journey",
            "connected in the system",
            "connected in system",
            "connected to the system",
            "linked in the system",
            "linked in system",
        )
    )


def _select_nodes_of_type(store, node_type: str, query: str | None = None, limit: int = 10) -> list[str]:
    candidates = [
        node_id
        for node_id, attrs in store.G.nodes(data=True)
        if attrs.get("node_type") == node_type
    ]
    if query:
        matches = _find_matching_nodes(store, query, node_type=node_type, limit=limit)
        if matches:
            return matches
    return sorted(candidates)[:limit]


def _is_clearly_off_topic(lowered: str) -> bool:
    return any(token in lowered for token in OFF_TOPIC_HINTS)


def _normalize_entity_query(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"'s\b", "s", normalized)
    normalized = re.sub(r"[']", "", normalized)
    normalized = re.sub(r"[?!.,;]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    filler_prefixes = (
        "tell me about",
        "tell about",
        "show me",
        "give me",
        "find",
        "lookup",
        "look up",
        "open",
    )
    changed = True
    while changed and normalized:
        changed = False
        for prefix in filler_prefixes:
            token = prefix + " "
            if normalized.startswith(token):
                normalized = normalized[len(token):].strip()
                changed = True
    return normalized


def _tokenize_normalized_query(text: str) -> list[str]:
    if not text:
        return []
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _has_token(tokens: list[str], options: set[str]) -> bool:
    return any(token in options for token in tokens)


def _has_phrase(text: str, phrases: tuple[str, ...] | list[str] | set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _query_mentions_customer(tokens: list[str], text: str) -> bool:
    if "customer" in tokens or "customers" in tokens:
        return True
    return any(
        phrase in text
        for phrase in (
            "this customer",
            "that customer",
            "the customer",
            "customer jordan",
            "customer fitzpatrick",
            "customer nelson",
            "customer bradley",
            "customer garner",
        )
    )


def _query_mentions_customer_orders(tokens: list[str], text: str) -> bool:
    return _query_mentions_customer(tokens, text) and _has_token(tokens, ORDER_REQUEST_TOKENS)


def _query_mentions_top_customers(tokens: list[str], text: str) -> bool:
    return _has_token(tokens, {"top", "most", "highest", "largest", "biggest"}) and _query_mentions_customer(tokens, text)


def _query_mentions_revenue_contributors(tokens: list[str], text: str) -> bool:
    revenue_like = _has_token(tokens, {"revenue", "revenues", "sales", "amount", "net"}) or "revenue" in text
    contributor_like = _has_token(tokens, {"top", "most", "highest", "largest", "biggest", "contribute", "contributes", "contributors"})
    return revenue_like and contributor_like


def _query_mentions_delivery_activity_customers(tokens: list[str], text: str) -> bool:
    delivery_like = _has_token(tokens, {"delivery", "deliveries"})
    activity_like = _has_token(tokens, {"activity", "activities", "linked", "connected", "top", "most", "highest", "largest", "biggest"})
    return delivery_like and activity_like and _query_mentions_customer(tokens, text)


def _needs_llm_repair(user_message: str, plan: dict[str, Any]) -> str | None:
    steps = plan.get("steps") or []
    first_action = steps[0].get("action") if steps else None
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    tokens = _tokenize_normalized_query(normalized)
    aggregate_tokens = {"top", "most", "highest", "largest", "biggest", "contribute", "contributors", "spend", "billing", "billed", "bill", "bills", "invoice", "invoices", "amount", "revenue"}
    relationship_tokens = RELATIONSHIP_REQUEST_TOKENS | {"connected"}
    aggregate_actions = {
        "customer_top_billed_orders",
        "customer_top_products",
        "aggregate_connected_sum",
        "aggregate_delivery_activity",
    }
    if first_action in aggregate_actions and _has_token(tokens, relationship_tokens) and not _has_token(tokens, aggregate_tokens):
        return "relationship"
    if first_action not in {"search_nodes", "get_node"}:
        return None
    if _has_token(tokens, {"random", "any", "some"}):
        return "random"
    if _has_token(tokens, {"orders", "order"}) and _has_token(tokens, {"associated", "association"}):
        return "orders_associated"
    if _has_token(tokens, aggregate_tokens):
        return "aggregate"
    if _has_token(tokens, relationship_tokens):
        return "relationship"
    if "this customer" in normalized or "that customer" in normalized:
        return "relationship"
    return None


def _query_is_billing_contributor(user_message: str) -> bool:
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    tokens = _tokenize_normalized_query(normalized)
    return _has_token(tokens, {"contribute", "contributors", "billing", "billed", "bill", "bills", "invoice", "invoices", "clients", "customers", "maximum", "biggest", "most"})

def _query_is_customer_billed_most(user_message: str) -> bool:
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    tokens = _tokenize_normalized_query(normalized)
    return _has_token(tokens, {"billed", "billing", "bill", "bills"}) and _has_token(tokens, {"most", "highest", "top"})

def _query_is_top_billed_order(user_message: str) -> bool:
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    tokens = _tokenize_normalized_query(normalized)
    return _has_token(tokens, {"order", "orders"}) and _has_token(tokens, {"billed", "billing", "bill", "bills"}) and _has_token(tokens, {"most", "highest", "top"})

def _query_is_top_products_by_billing_docs(user_message: str) -> bool:
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    tokens = _tokenize_normalized_query(normalized)
    return _has_token(tokens, {"product", "products"}) and _has_token(tokens, {"billing", "billed", "bill", "bills", "invoice", "invoices", "document", "documents"}) and _has_token(tokens, {"most", "highest", "top", "appear", "appears"})


def _plan_missing_customer_params(plan: dict[str, Any]) -> bool:
    steps = plan.get("steps") or []
    if not steps:
        return False
    step0 = steps[0]
    if not isinstance(step0, dict):
        return False
    action = step0.get("action")
    if action not in {"customer_top_billed_orders", "customer_top_products"}:
        return False
    params = step0.get("params") if isinstance(step0.get("params"), dict) else {}
    return not params.get("customer_id") and not params.get("customer_query")


def _query_mentions_product_billing_aggregate(tokens: list[str], text: str) -> bool:
    product_like = _has_token(tokens, {"product", "products"})
    billing_like = _has_token(tokens, {"billing", "invoice", "invoices", "document", "documents", "count", "counts"})
    return product_like and billing_like and _has_token(tokens, {"top", "most", "highest", "largest", "biggest", "appear", "appears", "appear in"})


def _query_mentions_flow_trace(tokens: list[str], text: str) -> bool:
    if not _has_token(tokens, FLOW_REQUEST_TOKENS):
        return False
    return _has_token(tokens, {"billing", "invoice", "document", "documents", "order", "orders", "customer", "customers", "payment", "payments", "journal", "entry", "entries"}) or _has_phrase(
        text,
        (
            "move through the system",
            "move through",
            "full flow",
            "trace the flow",
            "lifecycle",
            "how does",
            "how do",
        ),
    )


def _query_mentions_conditional(tokens: list[str], text: str) -> bool:
    return _has_token(tokens, CONDITIONAL_REQUEST_TOKENS) and _has_phrase(
        text,
        (
            "delivered but not billed",
            "not billed",
            "without payments",
            "without payment",
            "incomplete flow",
            "incomplete flows",
            "broken flow",
            "broken flows",
        ),
    )


def _resolve_entity_reference(
    node_type: str,
    user_message: str,
    conv_context: dict[str, Any],
    store,
    allow_random: bool = True,
) -> str | None:
    text = user_message.lower()
    pronoun_tokens = {"this", "that", "it", "its", "them", "those", "these", "same"}
    normalized = _normalize_entity_query(user_message)
    canonical = _canonicalize_keyword_tokens(normalized)
    entity_request = _extract_entity_request(canonical)

    direct_node_id = _extract_direct_node_id(user_message)
    if direct_node_id and store.has_node(direct_node_id):
        if store.G.nodes[direct_node_id].get("node_type") == node_type:
            return direct_node_id

    if node_type in {"Customer", "SalesOrder", "BillingDocument", "DeliveryDocument", "Product", "Plant", "BillingItem", "SalesOrderItem", "JournalEntry", "Payment"}:
        if any(token in text for token in pronoun_tokens):
            latest_match = _latest_entity_of_type(conv_context, node_type)
            if latest_match:
                return latest_match
            last_turn_matches = _last_turn_entities_of_type(conv_context, store, node_type)
            if last_turn_matches:
                return last_turn_matches[0]

    if entity_request["node_type"] == node_type:
        candidates = [
            node_id
            for node_id, attrs in store.G.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]
        if entity_request["query"]:
            candidates = _find_matching_nodes(store, entity_request["query"], node_type=node_type, limit=10)
        if not candidates:
            return None
        if entity_request["qualifier"] in ENTITY_QUALIFIER_TOKENS and allow_random:
            return random.choice(sorted(candidates))
        return sorted(candidates)[0]

    if _query_mentions_customer([token for token in canonical.split()], canonical) and node_type == "Customer":
        matches = _find_matching_nodes(store, canonical, node_type=node_type, limit=10)
        if matches:
            return matches[0]

    if node_type == "Customer":
        searchable_tokens = [
            token
            for token in canonical.split()
            if token not in QUERY_STOPWORDS
            and token not in ORDER_REQUEST_TOKENS
            and token not in INTENT_VERB_TOKENS
            and token not in RELATIONSHIP_REQUEST_TOKENS
            and token not in FLOW_REQUEST_TOKENS
        ]
        searchable_query = " ".join(searchable_tokens).strip()
        if searchable_query and not _query_mentions_customer([token for token in canonical.split()], canonical):
            matches = _find_matching_nodes(store, searchable_query, node_type=node_type, limit=10)
            if matches:
                return matches[0]

    return None


def _extract_customer_name_hint(normalized_query: str) -> str:
    tokens = _tokenize_normalized_query(normalized_query)
    if not tokens:
        return ""

    breakers = QUERY_STOPWORDS | ORDER_REQUEST_TOKENS | INTENT_VERB_TOKENS | {
        "customer",
        "customers",
        "for",
        "to",
        "from",
        "linked",
        "whose",
        "which",
        "who",
    }

    for idx, token in enumerate(tokens):
        if token not in {"customer", "customers"}:
            continue
        name_tokens: list[str] = []
        for next_token in tokens[idx + 1:]:
            if next_token in breakers:
                if name_tokens:
                    break
                continue
            name_tokens.append(next_token)
        if name_tokens:
            return " ".join(name_tokens).strip()
    return ""


def _resolve_customer_for_orders_intent(
    user_message: str,
    conv_context: dict[str, Any],
    store,
) -> str | None:
    lowered = user_message.lower()
    normalized = _normalize_entity_query(user_message)
    canonical = _canonicalize_keyword_tokens(normalized)

    # 1) Pronoun-based memory resolution should win for "this/that customer's orders".
    if ("customer" in canonical or "customers" in canonical) and any(
        token in canonical for token in ("this", "that", "it", "its", "same", "those", "these")
    ):
        latest_customer = _latest_entity_of_type(conv_context, "Customer")
        if latest_customer:
            return latest_customer
        last_turn_customers = _last_turn_entities_of_type(conv_context, store, "Customer")
        if last_turn_customers:
            return last_turn_customers[0]

    # 2) Direct id, if the user explicitly provided one.
    direct_node_id = _extract_direct_node_id(user_message)
    if direct_node_id and store.has_node(direct_node_id):
        if store.G.nodes[direct_node_id].get("node_type") == "Customer":
            return direct_node_id

    # 3) Explicit "customer <name>" extraction (e.g. customer nelson).
    customer_name_hint = _extract_customer_name_hint(canonical)
    if customer_name_hint:
        matches = _find_matching_nodes(store, customer_name_hint, node_type="Customer", limit=5)
        if matches:
            return matches[0]

    # 4) Fallback to generic customer resolver.
    resolved = _resolve_entity_reference("Customer", user_message, conv_context, store, allow_random=False)
    if resolved:
        return resolved

    # 5) Last-memory fallback if query still references "this customer".
    if "this customer" in lowered or "that customer" in lowered:
        latest_customer = _latest_entity_of_type(conv_context, "Customer")
        if latest_customer:
            return latest_customer
    return None


def _canonicalize_keyword_tokens(normalized: str) -> str:
    tokens = normalized.split()
    if not tokens:
        return normalized
    canonical_tokens: list[str] = []
    for token in tokens:
        if token in KEYWORD_LEXICON:
            canonical_tokens.append(token)
            continue
        if any(ch.isdigit() for ch in token) or ":" in token:
            canonical_tokens.append(token)
            continue
        cutoff = 0.82 if len(token) >= 6 else 0.75
        matches = get_close_matches(token, KEYWORD_LEXICON, n=1, cutoff=cutoff)
        canonical_tokens.append(matches[0] if matches else token)
    return " ".join(canonical_tokens)


def _extract_entity_request(normalized: str) -> dict[str, Any]:
    """
    Parse normalized request text into a loose entity-intent descriptor.
    Supports:
      - any/random/some + type
      - type + optional free text
      - generic node/entity requests
      - optional relationship intent
    """
    words = normalized.split()
    if not words:
        return {
            "qualifier": None,
            "node_type": None,
            "query": "",
            "is_generic_node": False,
            "wants_relationships": False,
        }

    wants_relationships = any(token in RELATIONSHIP_TOKENS for token in words)

    filtered = [w for w in words if w not in {"and", "all", "its", "their", "with", "of"}]
    qualifier = next((w for w in filtered if w in ENTITY_QUALIFIER_TOKENS), None)

    # Remove lightweight grammar tokens.
    filtered = [w for w in filtered if w not in ENTITY_ARTICLES and w not in ENTITY_QUALIFIER_TOKENS]

    # Generic node/entity route.
    if any(w in {"node", "entity"} for w in filtered):
        query_tokens = [w for w in filtered if w not in {"node", "entity"} and w not in RELATIONSHIP_TOKENS]
        return {
            "qualifier": qualifier,
            "node_type": None,
            "query": " ".join(query_tokens).strip(),
            "is_generic_node": True,
            "wants_relationships": wants_relationships,
        }

    # Typed entity route.
    node_type = _resolve_node_type_phrase(" ".join(filtered))
    if node_type:
        query_tokens = [w for w in filtered if w not in RELATIONSHIP_TOKENS]
        # Remove explicit type words from query tokens.
        type_words = set()
        for phrase, mapped_type in ENTITY_TYPE_ALIASES.items():
            if mapped_type == node_type:
                type_words.update(phrase.split())
        query_tokens = [w for w in query_tokens if w not in type_words]
        return {
            "qualifier": qualifier,
            "node_type": node_type,
            "query": " ".join(query_tokens).strip(),
            "is_generic_node": False,
            "wants_relationships": wants_relationships,
        }

    return {
        "qualifier": qualifier,
        "node_type": None,
        "query": "",
        "is_generic_node": False,
        "wants_relationships": wants_relationships,
    }


def _resolve_reference_node(user_message: str, conv_context: dict[str, Any], store) -> str | None:
    text = user_message.lower()
    if any(phrase in text for phrase in ("top one", "first one")) and conv_context["last_turn_nodes"]:
        return conv_context["last_turn_nodes"][0]

    expected_type = _resolve_expected_node_type(user_message)
    if expected_type:
        last_turn_matches = _last_turn_entities_of_type(conv_context, store, expected_type)
        if last_turn_matches:
            return last_turn_matches[0]
        latest_match = _latest_entity_of_type(conv_context, expected_type)
        if latest_match:
            return latest_match

    if any(token in text for token in ("this ", "that ", "same ", "it")) and conv_context["last_turn_nodes"]:
        return conv_context["last_turn_nodes"][0]
    return None


def _resolve_reference_nodes(user_message: str, conv_context: dict[str, Any], store, node_type: str) -> list[str]:
    text = user_message.lower()
    if any(phrase in text for phrase in ("those orders", "these orders", "of those orders", "which of those are delivered", "of those are delivered")):
        order_nodes = _last_turn_entities_of_type(conv_context, store, node_type)
        if order_nodes:
            return order_nodes
        latest_customer = _latest_entity_of_type(conv_context, "Customer")
        if latest_customer and latest_customer in store.G:
            customer_orders = [
                src
                for src, _, attrs in store.G.in_edges(latest_customer, data=True)
                if attrs.get("edge_type") == "SOLD_TO" and store.G.nodes[src].get("node_type") == "SalesOrder"
            ]
            if customer_orders:
                return sorted(dict.fromkeys(customer_orders))
    if any(phrase in text for phrase in ("those products", "these products")):
        product_nodes = _last_turn_entities_of_type(conv_context, store, node_type)
        if product_nodes:
            return product_nodes
    return []


def _is_complex_query(text: str) -> bool:
    """Return True if the query contains signals that require LLM interpretation."""
    lowered = text.lower()
    question_words = {"what", "which", "who", "how"}
    # Aggregation signals
    agg_signals = {"most", "top", "highest", "largest", "biggest", "contribute",
                   "contributors", "spend", "spending", "maximum", "mainly",
                   "mostly", "usually", "primary", "main", "appear in"}
    # Relationship / activity signals
    rel_signals = {"involved", "connected", "linked", "activity", "interact",
                   "associated", "association", "everything connected"}
    # Domain action/noun signals
    domain_actions = {"order", "orders", "billing", "billed", "invoice", "invoices",
                      "spend", "spent", "buy", "buys", "bought", "purchase", "purchased",
                      "product", "products", "payment", "payments", "delivery", "deliveries"}
    # Multi-step / reasoning signals
    reasoning_signals = {"which of those", "how many", "compare", "versus",
                         "difference between"}
    all_signals = agg_signals | rel_signals | reasoning_signals
    tokens = set(re.split(r"\s+", lowered))
    # Question-word + domain/action + entity mention → complex
    entity_mentioned = any(phrase in lowered for phrase in ("customer", "this customer", "that customer"))
    if (tokens & question_words) and (tokens & (agg_signals | rel_signals | domain_actions)) and entity_mentioned:
        return True
    # Check single-word signals
    if tokens & all_signals:
        return True
    # Check multi-word phrase signals
    phrase_signals = [
        "most billed", "highest billing", "top products", "top customers",
        "billing count", "billing activity", "spend the most",
        "billed the most", "billed most", "what does this customer",
        "what has this customer", "what is this customer",
        "everything connected", "all relationships",
        "which of those", "appear in the most",
        "contribute most", "generate maximum",
        "mainly billing", "main billing",
    ]
    for phrase in phrase_signals:
        if phrase in lowered:
            return True
    # If the query mentions a customer AND an action beyond simple lookup
    customer_mentioned = any(w in lowered for w in ("customer", "jordan", "nelson",
                                                     "fitzpatrick", "bradley", "garner",
                                                     "this customer", "that customer"))
    action_beyond_lookup = any(w in tokens for w in {"order", "orders", "billing",
                                                      "billed", "bought", "purchased",
                                                      "buy", "buys", "invoice", "invoices",
                                                      "product", "products", "payment",
                                                      "payments", "delivery", "deliveries"})
    if customer_mentioned and action_beyond_lookup:
        return True
    return False


def _build_direct_plan(user_message: str, conv_context: dict[str, Any], store) -> dict[str, Any] | None:
    """Deterministic planner: handles ONLY simple, unambiguous queries.
    Returns None for anything complex (caller should use LLM).
    """
    text = user_message.strip()
    lowered = text.lower()
    normalized = _normalize_entity_query(text)
    canonical_normalized = _canonicalize_keyword_tokens(normalized)
    canonical_tokens = _tokenize_normalized_query(canonical_normalized)

    # ── 0. Bail early if the query is complex → let LLM handle ──
    if _is_complex_query(text):
        return None

    if lowered in {"show me details", "show details", "details", "show me more"}:
        return {"intent": "off_topic", "message": "Please specify what entity you want details for (for example: customer name, sales order, or billing document)."}

    # Generic entity handling layer (must run before guardrails).
    direct_node_id = _extract_direct_node_id(text)
    if direct_node_id:
        if store.has_node(direct_node_id):
            return {
                "intent": "detail",
                "steps": [
                    {"action": "get_node", "params": {"node_id": direct_node_id}},
                ],
            }
        return {"intent": "not_found", "message": "No matching entity found in dataset"}

    # ── 1. Known conditional patterns (explicit phrasing only) ──
    if "invoices without payments" in lowered or "invoice without payments" in lowered:
        return {
            "intent": "conditional",
            "steps": [{"action": "find_invoices_without_payments", "params": {"limit": 100}}],
        }
    if "delivered but not billed" in lowered:
        return {
            "intent": "conditional",
            "steps": [{"action": "find_delivered_without_billing", "params": {}}],
        }
    if "incomplete flows" in lowered or "incomplete flow" in lowered:
        return {
            "intent": "conditional",
            "steps": [{"action": "find_incomplete_flows", "params": {"limit": 120}}],
        }

    # ── 2. Pronoun follow-ups binding to memory (simple detail/relationship) ──
    referenced_node_id = _resolve_reference_node(user_message, conv_context, store)
    if referenced_node_id and any(token in lowered for token in ("it", "its", "this", "that", "same", "them", "those")):
        wants_relationships = (
            _has_token(canonical_tokens, RELATIONSHIP_REQUEST_TOKENS)
            or _has_phrase(lowered, ("more details", "more info", "details", "relationships"))
        )
        steps = [{"action": "get_node", "params": {"node_id": referenced_node_id}}]
        if wants_relationships:
            steps.append({"action": "get_neighbours", "params": {"node_id": referenced_node_id, "direction": "both"}})
        return {"intent": "detail", "steps": steps}

    # ── 3. Unified entity lookup (simple entity requests only) ──
    entity_request = _extract_entity_request(canonical_normalized)

    def build_entity_steps(node_id: str, include_relationships: bool) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = [{"action": "get_node", "params": {"node_id": node_id}}]
        if include_relationships:
            steps.append({"action": "get_neighbours", "params": {"node_id": node_id, "direction": "both"}})
        return steps

    def select_from_candidates(candidates: list[str], qualifier: str | None) -> str:
        if not candidates:
            return ""
        ordered = sorted(candidates)
        if qualifier in ENTITY_QUALIFIER_TOKENS:
            return random.choice(ordered)
        return ordered[0]

    # Relationship-intent entity queries
    if entity_request["wants_relationships"] and (entity_request["is_generic_node"] or entity_request["node_type"]):
        if entity_request["node_type"]:
            typed_candidates = [
                node_id for node_id, attrs in store.G.nodes(data=True)
                if attrs.get("node_type") == entity_request["node_type"]
            ]
            if entity_request["query"]:
                typed_candidates = _find_matching_nodes(
                    store, entity_request["query"], node_type=entity_request["node_type"], limit=10
                )
            selected = select_from_candidates(typed_candidates, entity_request["qualifier"])
        else:
            generic_candidates = list(store.G.nodes())
            if entity_request["query"]:
                generic_candidates = _find_matching_nodes(store, entity_request["query"], node_type=None, limit=10)
            selected = select_from_candidates(generic_candidates, entity_request["qualifier"])
        if selected:
            return {"intent": "detail", "steps": build_entity_steps(selected, include_relationships=True)}
        return {"intent": "not_found", "message": "No matching entity found in dataset"}

    # Typed entity intents
    if entity_request["node_type"]:
        if entity_request["query"]:
            matches = _find_matching_nodes(
                store, entity_request["query"], node_type=entity_request["node_type"], limit=10
            )
            if matches:
                selected = select_from_candidates(matches, entity_request["qualifier"])
                return {"intent": "detail", "steps": build_entity_steps(selected, include_relationships=False)}
            return {"intent": "not_found", "message": "No matching entity found in dataset"}
        candidates = [
            node_id for node_id, attrs in store.G.nodes(data=True)
            if attrs.get("node_type") == entity_request["node_type"]
        ]
        selected = select_from_candidates(candidates, entity_request["qualifier"])
        if selected:
            return {"intent": "detail", "steps": build_entity_steps(selected, include_relationships=False)}
        return {"intent": "not_found", "message": "No matching entity found in dataset"}

    # Generic node intents
    if entity_request["is_generic_node"]:
        candidates = list(store.G.nodes())
        if entity_request["query"]:
            candidates = _find_matching_nodes(store, entity_request["query"], node_type=None, limit=10)
        selected = select_from_candidates(candidates, entity_request["qualifier"])
        if selected:
            return {"intent": "detail", "steps": build_entity_steps(selected, include_relationships=False)}
        return {"intent": "not_found", "message": "No matching entity found in dataset"}

    # ── 4. Fuzzy name search fallback ──
    if canonical_normalized:
        expected_type = _resolve_node_type_phrase(canonical_normalized)
        matches = _find_matching_nodes(store, canonical_normalized, node_type=expected_type, limit=10)
        if not matches and expected_type:
            matches = _find_matching_nodes(store, canonical_normalized, node_type=None, limit=10)
        if matches:
            return {
                "intent": "detail",
                "steps": [{"action": "get_node", "params": {"node_id": matches[0]}}],
            }

    # ── 5. Off-topic guardrail ──
    if _is_clearly_off_topic(lowered):
        return {"intent": "off_topic", "message": "This system is designed to answer dataset-related queries only."}

    # Nothing matched deterministically → return None so LLM handles it
    return None


def _is_dataset_related_query(user_message: str) -> bool:
    normalized = _canonicalize_keyword_tokens(_normalize_entity_query(user_message))
    if not normalized:
        return False
    if _extract_direct_node_id(user_message):
        return True
    tokens = _tokenize_normalized_query(normalized)
    if any(token in normalized for token in OFF_TOPIC_HINTS):
        return False
    if _resolve_node_type_phrase(normalized):
        return True
    domain_tokens = {
        "order",
        "orders",
        "sales",
        "delivery",
        "deliveries",
        "billing",
        "invoice",
        "invoices",
        "payment",
        "payments",
        "journal",
        "entry",
        "revenue",
        "activity",
        "activities",
        "customer",
        "customers",
        "product",
        "products",
        "plant",
        "flow",
        "trace",
        "lifecycle",
        "top",
        "most",
        "delivered",
        "billed",
        "incomplete",
        "relationships",
        "relationship",
        "connected",
        "linked",
        "node",
        "entity",
    }
    return any(token in domain_tokens for token in tokens)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = text[start:end + 1]
    try:
        parsed = json.loads(fragment)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _repair_node_id(node_id: str, store) -> str | None:
    if not node_id:
        return None
    if store.has_node(node_id):
        return node_id
    if ":" not in node_id:
        return None
    node_type, raw_value = node_id.split(":", 1)
    if node_type not in VALID_NODE_TYPES:
        return None
    query = raw_value.replace(":", " ").replace("_", " ").replace("-", " ").strip()
    if not query:
        return None
    matches = _find_matching_nodes(store, query, node_type=node_type, limit=1)
    return matches[0] if matches else None


_DEFAULT_GROUP_BY: dict[str, str] = {
    "Customer": "name",
    "Product": "description",
    "SalesOrder": "salesOrder_id",
    "DeliveryDocument": "deliveryDocument_id",
    "BillingDocument": "billingDocument_id",
    "BillingItem": "billingItem_id",
    "JournalEntry": "journalEntry_id",
    "Plant": "plantName",
}


def _pick_group_by_for_node_type(node_type: str | None, store) -> str | None:
    if not node_type:
        return None
    default = _DEFAULT_GROUP_BY.get(node_type)
    if default:
        return default
    for node_id, attrs in store.G.nodes(data=True):
        if attrs.get("node_type") != node_type:
            continue
        for field in (
            "name",
            "description",
            "label",
            "customer_id",
            "product_id",
            "billingDocument_id",
            "billingItem_id",
            "salesOrder_id",
            "deliveryDocument_id",
            "journalEntry_id",
            "plant_id",
        ):
            if field in attrs:
                return field
    return None


def _normalize_llm_aggregate_params(params: dict[str, Any], store) -> dict[str, Any] | None:
    node_type = params.get("node_type")
    if node_type and node_type not in VALID_NODE_TYPES:
        return None

    normalized = dict(params)
    relation_tokens = " ".join(
        str(value)
        for key, value in params.items()
        if key in {"relation", "relation_type", "aggregation"} and value
    ).lower()

    if relation_tokens and node_type:
        if node_type == "Customer" and any(token in relation_tokens for token in ("billing", "invoice", "billingdocument")):
            normalized["count_connected_node_type"] = "BillingDocument"
            normalized["via_edge_type"] = "BILLED_TO"
        elif node_type == "Customer" and any(token in relation_tokens for token in ("order", "salesorder")):
            normalized["count_connected_node_type"] = "SalesOrder"
            normalized["via_edge_type"] = "SOLD_TO"
        elif node_type == "Product" and any(token in relation_tokens for token in ("billing", "invoice", "billingdocument")):
            normalized["count_connected_node_type"] = "BillingItem"
            normalized["via_edge_type"] = "BILLS_PRODUCT"
        elif node_type == "BillingDocument" and any(token in relation_tokens for token in ("billing", "billed", "items", "item")):
            normalized["count_connected_node_type"] = "BillingItem"
            normalized["via_edge_type"] = "HAS_ITEM"
        elif node_type == "Customer" and any(token in relation_tokens for token in ("revenue", "amount", "sales")):
            normalized["sum_connected_node_type"] = "BillingDocument"
            normalized["sum_connected_field"] = "totalNetAmount"
            normalized["via_edge_type"] = "BILLED_TO"
            normalized["_action_override"] = "aggregate_connected_sum"
        elif node_type == "Customer" and any(token in relation_tokens for token in ("delivery", "deliveries")):
            normalized["_action_override"] = "aggregate_delivery_activity"

    if normalized.get("count_connected_node_type") and not normalized.get("via_edge_type"):
        count_type = normalized.get("count_connected_node_type")
        if node_type == "Customer" and count_type == "BillingDocument":
            normalized["via_edge_type"] = "BILLED_TO"
        elif node_type == "Customer" and count_type == "SalesOrder":
            normalized["via_edge_type"] = "SOLD_TO"
        elif node_type == "Product" and count_type in {"BillingItem", "BillingDocument"}:
            normalized["via_edge_type"] = "BILLS_PRODUCT"

    if not normalized.get("group_by"):
        fallback_group_by = _pick_group_by_for_node_type(node_type, store)
        if fallback_group_by:
            normalized["group_by"] = fallback_group_by

    allowed_keys = {
        "node_type",
        "group_by",
        "metric",
        "metric_field",
        "top_n",
        "via_edge_type",
        "count_connected_node_type",
        "sum_connected_node_type",
        "sum_connected_field",
        "_action_override",
    }
    for key in list(normalized.keys()):
        if key not in allowed_keys:
            normalized.pop(key, None)

    return normalized


def _sanitize_llm_plan(plan: dict[str, Any], store) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    if plan.get("intent") == "off_topic":
        return {"intent": "off_topic", "message": "This system is designed to answer dataset-related queries only."}

    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    sanitized_steps: list[dict[str, Any]] = []
    for step in raw_steps[:4]:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip()
        if action not in ALLOWED_ACTIONS:
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            params = {}
        safe_params = dict(params)

        if "node_id" in safe_params and isinstance(safe_params["node_id"], str):
            repaired = _repair_node_id(safe_params["node_id"], store)
            if repaired is None:
                continue
            safe_params["node_id"] = repaired
        elif action in {"get_node", "get_neighbours", "find_connected"}:
            alt_id = safe_params.get("id")
            alt_type = safe_params.get("type") or safe_params.get("node_type")
            if isinstance(alt_id, str) and isinstance(alt_type, str):
                candidate = f"{alt_type}:{alt_id}"
                repaired = _repair_node_id(candidate, store)
                if repaired is None:
                    continue
                safe_params["node_id"] = repaired
            else:
                continue

        if action == "get_subgraph":
            center = safe_params.get("center")
            if isinstance(center, str):
                repaired = _repair_node_id(center, store)
                if repaired is None:
                    continue
                safe_params["center"] = repaired

        if action == "trace_flow":
            node_id = safe_params.get("node_id")
            if not isinstance(node_id, str) or not store.has_node(node_id):
                continue
            safe_params["max_depth"] = min(max(int(safe_params.get("max_depth", 6)), 1), 6)

        if action == "find_connected":
            node_id = safe_params.get("node_id")
            if not isinstance(node_id, str):
                continue
            repaired = _repair_node_id(node_id, store)
            if repaired is None:
                continue
            safe_params["node_id"] = repaired
            safe_params["max_depth"] = min(max(int(safe_params.get("max_depth", 3)), 1), 5)

        if action in {"customer_top_billed_orders", "customer_top_products"}:
            customer_id = safe_params.get("customer_id")
            customer_query = safe_params.get("customer_query") or safe_params.get("customer_name") or safe_params.get("name") or safe_params.get("customer")
            if customer_id and isinstance(customer_id, str):
                if ":" not in customer_id:
                    candidate = f"Customer:{customer_id}"
                    repaired = _repair_node_id(candidate, store)
                    if repaired:
                        customer_id = repaired
                    else:
                        customer_query = customer_query or customer_id
                elif store.has_node(customer_id):
                    pass
                else:
                    repaired = _repair_node_id(customer_id, store)
                    if repaired:
                        customer_id = repaired
                    else:
                        customer_query = customer_query or customer_id
            if not customer_id and customer_query:
                safe_params["customer_query"] = str(customer_query)
            elif customer_id:
                safe_params["customer_id"] = customer_id
            else:
                continue

        if action == "aggregate":
            safe_params["top_n"] = min(max(int(safe_params.get("top_n", 10)), 1), 25)
            normalized = _normalize_llm_aggregate_params(safe_params, store)
            if normalized is None:
                continue
            action_override = normalized.pop("_action_override", None)
            if action_override in {"aggregate_connected_sum", "aggregate_delivery_activity"}:
                action = action_override
            safe_params = normalized

        if action == "aggregate_connected_sum":
            node_type = safe_params.get("node_type")
            if node_type and node_type not in VALID_NODE_TYPES:
                continue
            safe_params["top_n"] = min(max(int(safe_params.get("top_n", 10)), 1), 25)
            if not safe_params.get("group_by"):
                fallback_group_by = _pick_group_by_for_node_type(node_type, store)
                if fallback_group_by:
                    safe_params["group_by"] = fallback_group_by
            if not (safe_params.get("via_edge_type") and safe_params.get("sum_connected_node_type") and safe_params.get("sum_connected_field")):
                continue

        if action == "aggregate_delivery_activity":
            safe_params = {"top_n": min(max(int(safe_params.get("top_n", 10)), 1), 25)}

        if action == "random_node":
            node_type = safe_params.get("node_type")
            if node_type and node_type not in VALID_NODE_TYPES:
                continue
            safe_params["limit"] = min(max(int(safe_params.get("limit", 1)), 1), 5)

        if action in {"get_nodes_by_ids", "filter_node_ids"}:
            node_ids = safe_params.get("node_ids")
            if not isinstance(node_ids, list):
                continue
            repaired_ids = []
            for value in node_ids[:100]:
                if not isinstance(value, str):
                    continue
                repaired = _repair_node_id(value, store)
                if repaired:
                    repaired_ids.append(repaired)
            if not repaired_ids:
                continue
            safe_params["node_ids"] = repaired_ids

        if action == "search_nodes":
            query = safe_params.get("query", "")
            if not isinstance(query, str) or not query.strip():
                continue
            if "node_type" not in safe_params and isinstance(safe_params.get("node_types"), list):
                node_types = [t for t in safe_params.get("node_types") if t in VALID_NODE_TYPES]
                if len(node_types) == 1:
                    safe_params["node_type"] = node_types[0]
            safe_params["limit"] = min(max(int(safe_params.get("limit", 20)), 1), 50)

        sanitized_steps.append({"action": action, "params": safe_params})

    if not sanitized_steps:
        return None
    return {"intent": str(plan.get("intent", "detail")), "steps": sanitized_steps}


def _normalize_llm_plan_for_context(
    plan: dict[str, Any],
    user_message: str,
    conv_context: dict[str, Any],
    store,
) -> dict[str, Any]:
    return plan


def _build_not_found_reason(user_message: str, store) -> str:
    expected_type = _resolve_node_type_phrase(_normalize_entity_query(user_message)) or _resolve_expected_node_type(user_message)
    if expected_type:
        filtered_tokens = store.filter_query_tokens(user_message, node_type=expected_type)
        if not filtered_tokens:
            return f"Reason: no {expected_type} identifiers from the query were found in the dataset."
        token_preview = ", ".join(filtered_tokens[:4])
        return f"Reason: no {expected_type} matched tokens ({token_preview})."
    filtered_tokens = store.filter_query_tokens(user_message, node_type=None)
    if filtered_tokens:
        token_preview = ", ".join(filtered_tokens[:4])
        return f"Reason: no entities matched tokens ({token_preview})."
    return "Reason: could not infer a valid entity or matching identifiers from the query."


def _build_llm_context(conv_context: dict[str, Any], max_items: int = 8) -> str:
    if not conv_context.get("ordered_entities"):
        return "No prior entity context."
    recent = conv_context["ordered_entities"][-max_items:]
    rows = []
    for item in recent:
        rows.append(f"- {item['node_id']} ({item['node_type']}): {item.get('label', '')}")
    return "\n".join(rows)


def _call_llm_query_planner(
    user_message: str,
    conv_context: dict[str, Any],
    store,
    mode: str = "primary",
    hint: str | None = None,
) -> dict[str, Any] | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and not openrouter_key:
        return None

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
    planner_prompt = (
        "You are a query planner for a SAP O2C graph. "
        "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
        "Allowed actions: get_node, get_nodes_by_ids, get_neighbours, get_subgraph, search_nodes, count_by_type, "
        "filter_nodes, filter_node_ids, aggregate, aggregate_connected_sum, aggregate_delivery_activity, "
        "top_billed_orders, top_products_by_billing_documents, customer_top_billed_orders, customer_top_products, random_node, trace_flow, find_connected, "
        "find_delivered_without_billing, find_billed_without_delivery, find_invoices_without_payments, "
        "find_incomplete_flows, stats. "
        "Always prefer using conversation_entities for 'this/that/it/their' references. "
        "For name queries (e.g., 'customer jordan'), use search_nodes with the name. "
        "For relationship questions (involved/linked/connected), use get_neighbours with a concrete node_id. "
        "For 'most billed order for this customer', use customer_top_billed_orders with customer_id. "
        "For 'billed most for' phrasing, use customer_top_products with customer_id. "
        "For 'orders associated with a customer', prefer customer_top_billed_orders if billing is mentioned; otherwise use get_neighbours with edge_type SOLD_TO. "
        "For 'spend most on' questions, use customer_top_products with customer_id. "
        "For 'customers contribute most to billing', use aggregate_connected_sum on Customer via BILLED_TO summing totalNetAmount. "
        "For 'most billed order' without a customer, use top_billed_orders. "
        "For 'products associated with the highest number of billing documents', use top_products_by_billing_documents. "
        "For random entity requests, use random_node with node_type when specified. "
        "Use exact node IDs when available from context; otherwise create search_nodes/filter/aggregate/conditional plans. "
        "If unsure, default to search_nodes with the raw user query. "
        "Do not output markdown."
    )
    if mode == "repair":
        planner_prompt += " The previous plan failed. Ensure you return a valid plan using available actions and conversation_entities."
    if mode == "search_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return a search_nodes plan with params {\"query\": <user_query>, \"limit\": 10}. "
            "Do not output markdown."
        )
    if mode == "relationship_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return a get_neighbours plan using a concrete node_id from conversation_entities. "
            "Example: {\"intent\":\"relationship\",\"steps\":[{\"action\":\"get_neighbours\",\"params\":{\"node_id\":\"Customer:...\",\"direction\":\"both\",\"limit\":25}}]}. "
            "Do not output markdown."
        )
    if mode == "customer_billed_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return a customer_top_products plan with params {\"customer_query\": <customer_name_from_user>, \"top_n\": 5}. "
            "Use the name from the user query. Do not output markdown."
        )
    if mode == "top_billed_orders_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return a top_billed_orders plan with params {\"top_n\": 10}. "
            "Do not output markdown."
        )
    if mode == "top_products_by_billing_docs_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return a top_products_by_billing_documents plan with params {\"top_n\": 10}. "
            "Do not output markdown."
        )
    if mode == "aggregate_fallback":
        planner_prompt = (
            "Return ONLY JSON with shape {\"intent\": string, \"steps\": [{\"action\": string, \"params\": object}]}. "
            "You must return an aggregate_connected_sum plan for top billing contributors: "
            "{\"action\":\"aggregate_connected_sum\",\"params\":{\"node_type\":\"Customer\",\"group_by\":\"name\","
            "\"via_edge_type\":\"BILLED_TO\",\"sum_connected_node_type\":\"BillingDocument\","
            "\"sum_connected_field\":\"totalNetAmount\",\"top_n\":10}}. "
            "Do not output markdown."
        )
    if hint:
        if hint == "aggregate":
            planner_prompt += " Repair hint: if the query mentions orders + a customer name, return customer_top_billed_orders with customer_query set to the name. If it mentions spend/billed most, return customer_top_products with customer_query. Avoid search_nodes."
        elif hint == "relationship":
            planner_prompt += " Repair hint: for phrases like connected/linked/involved/interact/activity, always use get_neighbours with a concrete node_id from conversation_entities. Do not use aggregation."
        elif hint == "random":
            planner_prompt += " Repair hint: use random_node with a node_type if specified."
        elif hint == "orders_associated":
            planner_prompt += " Repair hint: return customer_top_billed_orders with customer_query set to the customer name from the query."
        else:
            planner_prompt += f" Repair hint: {hint}"
    user_payload = {
        "query": user_message,
        "conversation_entities": _build_llm_context(conv_context),
        "known_node_types": sorted(list(VALID_NODE_TYPES)),
    }
    if hint:
        user_payload["repair_hint"] = hint
    request_body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    }
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    groq_req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        if api_key:
            with urllib.request.urlopen(groq_req, timeout=6, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        else:
            payload = None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        logger.warning(f"LLM planner HTTP error status={exc.code} body={body[:400]}")
        if exc.code in {429, 402, 403}:
            payload = None
        else:
            return None
    except Exception as exc:
        logger.warning(f"LLM planner request failed: {exc}")
        payload = None

    if payload:
        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = _extract_json_object(content)
            if parsed:
                return _sanitize_llm_plan(parsed, store)
        except Exception:
            pass

    if not openrouter_key:
        return {"intent": "service_unavailable", "message": "Service temporarily unavailable due to usage limits"}

    openrouter_body = {
        "model": openrouter_model,
        "temperature": 0.0,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    }
    openrouter_req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(openrouter_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Dodge AI O2C",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(openrouter_req, timeout=8, context=ssl_context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        logger.warning(f"OpenRouter planner HTTP error status={exc.code} body={body[:400]}")
        return {"intent": "service_unavailable", "message": "Service temporarily unavailable due to usage limits"}
    except Exception as exc:
        logger.warning(f"OpenRouter planner request failed: {exc}")
        return {"intent": "service_unavailable", "message": "Service temporarily unavailable due to usage limits"}

    try:
        content = payload["choices"][0]["message"]["content"]
    except Exception:
        return None
    parsed = _extract_json_object(content)
    if not parsed:
        return None
    return _sanitize_llm_plan(parsed, store)


def _summarize_node(node: dict[str, Any], store) -> str:
    node_id = node.get("id", "UNKNOWN")
    node_type = node.get("node_type", node_id.split(":", 1)[0])
    fields = []
    preferred_keys = [
        "name",
        "description",
        "customer_id",
        "product_id",
        "salesOrder_id",
        "billingDocument_id",
        "deliveryDocument_id",
        "accountingDocument_id",
        "city",
        "country",
        "region",
        "totalNetAmount",
        "transactionCurrency",
        "overallDeliveryStatus",
        "billingDocumentDate",
        "postingDate",
        "clearingDate",
    ]
    for key in preferred_keys:
        if key in node and node[key] not in (None, "", []):
            fields.append(f"{key}: {node[key]}")
    fields_str = "\n".join(f"- {field}" for field in fields[:8])
    return f"**{node_type}** `{node_id}`\n{fields_str}" if fields_str else f"**{node_type}** `{node_id}`"


def _summarize_execution_result(user_message: str, plan: dict[str, Any], execution_result: dict[str, Any], store) -> str:
    data = execution_result.get("data", {})
    step0 = data.get("step_0")
    steps = plan.get("steps", [])
    first_action = steps[0].get("action") if steps else None

    if first_action == "get_node" and isinstance(step0, dict):
        parts = [_summarize_node(step0, store)]
        step1 = data.get("step_1")
        if isinstance(step1, dict) and "neighbours" in step1:
            neighbours = step1.get("neighbours", [])
            if neighbours:
                rendered = [f"- `{item['id']}` ({item.get('node_type', 'UNKNOWN')}) via {item.get('edge_type', 'UNKNOWN')}" for item in neighbours[:12]]
                if len(neighbours) > 12:
                    rendered.append(f"- ... and {len(neighbours) - 12} more")
                parts.append("**Connected entities**\n" + "\n".join(rendered))
        return "\n\n".join(parts)

    if first_action == "get_neighbours" and isinstance(step0, dict):
        neighbours = step0.get("neighbours", [])
        if not neighbours:
            return "No connected entities found."
        heading = "**Connected entities**"
        if all(item.get("node_type") == "SalesOrder" for item in neighbours):
            heading = f"**Sales orders for** `{step0.get('node_id', 'UNKNOWN')}`"
        elif all(item.get("node_type") == "Customer" for item in neighbours):
            heading = f"**Customers connected to** `{step0.get('node_id', 'UNKNOWN')}`"
        rendered = [f"- `{item['id']}` ({item.get('node_type', 'UNKNOWN')})" for item in neighbours[:25]]
        if len(neighbours) > 25:
            rendered.append(f"- ... and {len(neighbours) - 25} more")
        return heading + "\n" + "\n".join(rendered)

    if first_action == "filter_node_ids" and isinstance(step0, dict):
        results = step0.get("results", [])
        if not results:
            return "No matching entities found in the previous result set."
        rendered = [f"- `{item['id']}`" for item in results[:25]]
        if len(results) > 25:
            rendered.append(f"- ... and {len(results) - 25} more")
        return f"**Matching prior-turn entities**\n" + "\n".join(rendered)

    if first_action in {"aggregate", "aggregate_connected_sum", "aggregate_delivery_activity", "top_billed_orders", "top_products_by_billing_documents", "customer_top_billed_orders", "customer_top_products"} and isinstance(step0, dict):
        rows = step0.get("aggregation", [])
        if not rows:
            return "No aggregate results found."
        rendered = []
        for row in rows[:10]:
            label = row.get("label") or row.get("group") or row.get("node_id", "UNKNOWN")
            node_id = row.get("node_id")
            suffix = f" `{node_id}`" if node_id else ""
            if "value" in row:
                rendered.append(f"- {label}{suffix}: {row.get('value', 0)}")
            else:
                rendered.append(f"- {label}{suffix}: {row.get('count', 0)}")
        return "**Top results**\n" + "\n".join(rendered)

    if first_action in {"search_nodes", "filter_nodes", "random_node"} and isinstance(step0, dict):
        results = step0.get("results", [])
        if not results:
            return "No matching entity found in dataset"
        if len(results) == 1:
            return _summarize_node(results[0], store)
        rendered = [
            f"- `{item.get('id', 'UNKNOWN')}` ({item.get('node_type', 'UNKNOWN')})"
            for item in results[:10]
        ]
        if len(results) > 10:
            rendered.append(f"- ... and {len(results) - 10} more")
        return "**Closest matches**\n" + "\n".join(rendered)

    if first_action == "get_nodes_by_ids" and isinstance(step0, dict):
        nodes = step0.get("results", [])
        if not nodes:
            return "No matching entity found in dataset"
        rendered = [
            f"- `{item.get('id', 'UNKNOWN')}` ({item.get('node_type', 'UNKNOWN')})"
            for item in nodes[:10]
        ]
        return "**Closest matches**\n" + "\n".join(rendered)

    if first_action in {"find_delivered_without_billing", "find_billed_without_delivery", "find_invoices_without_payments", "find_incomplete_flows"} and isinstance(step0, dict):
        rows = step0.get("results", [])
        if not rows:
            return "No matching entities found."
        rendered = []
        for item in rows[:30]:
            suffix = ""
            missing = item.get("missing_stages")
            if isinstance(missing, list) and missing:
                suffix = f" missing: {', '.join(missing)}"
            rendered.append(f"- `{item.get('id', 'UNKNOWN')}` ({item.get('node_type', 'UNKNOWN')}){suffix}")
        if len(rows) > 30:
            rendered.append(f"- ... and {len(rows) - 30} more")
        return "**Matching entities**\n" + "\n".join(rendered)

    if first_action == "trace_flow" and isinstance(step0, dict):
        nodes = step0.get("flow_nodes", [])
        edges = step0.get("flow_edges", [])
        if not nodes:
            return "No flow nodes found."
        rendered = [f"- `{n.get('id', 'UNKNOWN')}` ({n.get('node_type', 'UNKNOWN')})" for n in nodes[:25]]
        if len(nodes) > 25:
            rendered.append(f"- ... and {len(nodes) - 25} more nodes")
        header = f"**Flow trace**\nNodes: {len(nodes)} · Edges: {len(edges)}"
        if step0.get("truncated"):
            header += "\nShowing top results only due to graph traversal limits."
        return header + "\n" + "\n".join(rendered)

    return json.dumps(execution_result.get("data", {}), indent=2, default=str)


def _extract_referenced_nodes(
    user_message: str,
    plan: dict[str, Any],
    execution_result: dict[str, Any],
    result_str: str,
    store,
    limit: int = 50,
) -> list[str]:
    """Extract node ids from execution output, including aggregate labels when ids are absent."""
    referenced_nodes: list[str] = []
    seen: set[str] = set()

    def add_node_id(node_id: str) -> None:
        if node_id in seen:
            return
        if not store.has_node(node_id):
            return
        seen.add(node_id)
        referenced_nodes.append(node_id)

    # 1) Direct NodeType:Value id extraction.
    try:
        matches = re.findall(r'"([a-zA-Z]+:[a-zA-Z0-9_\-:]+)"', result_str)
        for nid in matches:
            add_node_id(nid)
            if len(referenced_nodes) >= limit:
                return referenced_nodes
    except Exception as e:
        logger.warning(f"Failed to extract direct node IDs: {e}")

    data = execution_result.get("data", {})

    # 2) Structured id fields inside execution payload.
    def walk_for_node_ids(value: Any) -> None:
        if len(referenced_nodes) >= limit:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"id", "node_id"} and isinstance(nested, str):
                    add_node_id(nested)
                walk_for_node_ids(nested)
        elif isinstance(value, list):
            for item in value:
                walk_for_node_ids(item)

    walk_for_node_ids(data)
    if len(referenced_nodes) >= limit:
        return referenced_nodes

    # 3) Aggregate labels (e.g. top customers/products) -> resolve to node IDs.
    expected_type = _resolve_expected_node_type(user_message)
    candidate_labels: list[str] = []

    def walk_for_labels(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_lower = key.lower()
                if isinstance(nested, str) and key_lower in {
                    "label",
                    "group",
                    "name",
                    "description",
                    "customer",
                    "customer_name",
                    "product",
                    "product_name",
                    "value",
                }:
                    candidate_labels.append(nested.strip())
                walk_for_labels(nested)
        elif isinstance(value, list):
            for item in value:
                walk_for_labels(item)

    walk_for_labels(data)
    first_action = (plan.get("steps") or [{}])[0].get("action")
    if first_action in {"aggregate", "aggregate_connected_sum", "aggregate_delivery_activity", "top_billed_orders", "top_products_by_billing_documents", "customer_top_billed_orders", "customer_top_products"}:
        for row in data.get("step_0", {}).get("aggregation", []) if isinstance(data, dict) else []:
            label = row.get("label") or row.get("group")
            if isinstance(label, str):
                candidate_labels.append(label.strip())

    for label in dict.fromkeys(candidate_labels):
        if len(referenced_nodes) >= limit:
            break
        if not label or len(label) < 3:
            continue
        if NODE_ID_PATTERN.search(label):
            continue
        matches = _find_matching_nodes(
            store,
            label,
            node_type=expected_type,
            limit=1 if expected_type else 3,
        )
        for node_id in matches:
            add_node_id(node_id)
            if len(referenced_nodes) >= limit:
                break

    # 4) Search/filter results may already contain ids.
    first_action = (plan.get("steps") or [{}])[0].get("action")
    if first_action in {"search_nodes", "filter_nodes", "random_node"}:
        for item in data.get("step_0", {}).get("results", []) if isinstance(data, dict) else []:
            node_id = item.get("id")
            if node_id:
                add_node_id(node_id)
            if len(referenced_nodes) >= limit:
                break

    return referenced_nodes


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("")
def ask_assistant(request: Request, body: ChatRequest) -> Any:
    """Process a natural language query through Deterministic → LLM → Execute → Summarise pipeline."""
    request_started = time.monotonic()
    logger.info("PIPELINE START")
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    store = request.app.state.store
    engine = QueryEngine(store)

    user_message = body.messages[-1].get("content", "")
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Last message has no content")
    logger.info(f"USER MESSAGE: {user_message}")

    # ----- Step 0: Build conversation context from prior turns -----
    logger.info("MEMORY BUILD START")
    conv_context = _build_conversation_context(body, store)
    logger.info("MEMORY BUILD END")

    # ----- Step 1: Try deterministic plan first -----
    logger.info("PLANNER START")
    plan = _build_direct_plan(user_message, conv_context, store)

    # ----- Step 2: If deterministic returned None or not_found, try LLM -----
    needs_llm = (
        plan is None
        or (plan.get("intent") == "not_found" and _is_dataset_related_query(user_message))
    )
    if needs_llm:
        logger.info("LLM PLANNER START")
        llm_plan = _call_llm_query_planner(user_message, conv_context, store)
        logger.info("LLM PLANNER END")
        if llm_plan and llm_plan.get("intent") == "service_unavailable":
            # LLM unavailable — if we had a deterministic plan, use it; otherwise show error
            if plan and plan.get("intent") != "not_found":
                pass  # keep deterministic plan
            else:
                plan = {
                    "intent": "off_topic",
                    "message": llm_plan.get("message", "Service temporarily unavailable due to usage limits"),
                }
        elif llm_plan and llm_plan.get("intent") != "off_topic":
            plan = llm_plan
        elif llm_plan and llm_plan.get("intent") == "off_topic":
            # LLM says off_topic — only trust it if query is clearly off-topic
            lowered = user_message.lower()
            if _is_clearly_off_topic(lowered):
                plan = llm_plan
            elif plan is None:
                # LLM says off_topic but it's not clearly off_topic — try one repair
                repair_plan = _call_llm_query_planner(user_message, conv_context, store, mode="repair")
                if repair_plan and repair_plan.get("intent") not in {"off_topic", "service_unavailable", None}:
                    plan = repair_plan
                else:
                    plan = {"intent": "not_found", "message": "No matching entity found in dataset"}
        else:
            # LLM returned None — keep deterministic result or not_found
            if plan is None:
                plan = {"intent": "not_found", "message": "No matching entity found in dataset"}

    # ----- Step 2a: LLM fallback for specific aggregate intents -----
    steps = plan.get("steps") or []
    first_action = steps[0].get("action") if steps else None
    if _query_is_top_billed_order(user_message) and first_action in {"search_nodes", "get_node"}:
        billed_order_plan = _call_llm_query_planner(user_message, conv_context, store, mode="top_billed_orders_fallback")
        if billed_order_plan:
            plan = billed_order_plan
    steps = plan.get("steps") or []
    first_action = steps[0].get("action") if steps else None
    if _query_is_top_products_by_billing_docs(user_message) and first_action in {"search_nodes", "get_node"}:
        top_products_plan = _call_llm_query_planner(user_message, conv_context, store, mode="top_products_by_billing_docs_fallback")
        if top_products_plan:
            plan = top_products_plan

    # If plan is still None at this point (shouldn't happen, but safety)
    if plan is None:
        plan = {"intent": "not_found", "message": "No matching entity found in dataset"}

    # ----- Step 2b: Fix missing customer_id in customer_top_* plans from LLM -----
    steps = plan.get("steps") or []
    if steps:
        first_step = steps[0] if isinstance(steps[0], dict) else {}
        action = first_step.get("action", "")
        params = first_step.get("params", {}) if isinstance(first_step.get("params"), dict) else {}
        if action in {"customer_top_billed_orders", "customer_top_products"} and not params.get("customer_id") and not params.get("customer_query"):
            # Try to fill from conversation context
            last_customer = _latest_entity_of_type(conv_context, "Customer")
            if last_customer:
                params["customer_id"] = last_customer
                first_step["params"] = params

    logger.info("PLANNER END")

    def instant_stream(message: str):
        yield f'data: {json.dumps({"token": message})}\n\n'
        yield f'data: {json.dumps({"referenced_nodes": []})}\n\n'
        logger.info("STREAM END")
        yield 'data: [DONE]\n\n'

    def instant_stream_with_nodes(message: str, node_ids: list[str]):
        yield f'data: {json.dumps({"token": message})}\n\n'
        yield f'data: {json.dumps({"referenced_nodes": node_ids})}\n\n'
        logger.info(f"STREAM END nodes={len(node_ids)} elapsed_ms={int((time.monotonic()-request_started)*1000)}")
        yield 'data: [DONE]\n\n'

    logger.info(f"QUERY GENERATED: {json.dumps(plan, default=str)}")

    # ----- Step 3: Handle off-topic / unresolved -----
    if plan.get("intent") in {"off_topic", "not_found"}:
        message = plan.get("message", "That question is outside my scope. I can only help with SAP O2C data.")
        if plan.get("intent") == "not_found":
            reason = _build_not_found_reason(user_message, store)
            message = f"{message}\n{reason}"
        return StreamingResponse(instant_stream(message), media_type="text/event-stream")

    # ----- Step 4: Execute the plan against the graph -----
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        logger.info("EXECUTION START")
        future = executor.submit(engine.execute_plan, plan)
        execution_result = future.result(timeout=8.0)
        logger.info("EXECUTION END")
    except FuturesTimeoutError:
        logger.exception("Execution timeout")
        future.cancel()
        return StreamingResponse(instant_stream("This query is complex. Showing top results only is not available right now. Please try a narrower query."), media_type="text/event-stream")
    except Exception as e:
        logger.exception("Plan execution failed")
        return StreamingResponse(instant_stream(f"Query execution error: {e}"), media_type="text/event-stream")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # ----- Step 5: Summarize results -----
    logger.info("SUMMARY START")
    result_str = json.dumps(execution_result.get("data", {}), default=str)
    logger.info(f"RESULT SIZE chars={len(result_str)}")

    referenced_nodes = _extract_referenced_nodes(
        user_message=user_message,
        plan=plan,
        execution_result=execution_result,
        result_str=result_str,
        store=store,
        limit=50,
    )

    if len(result_str) > 12000:
        result_str = result_str[:12000] + "\n... [truncated]"

    local_summary = _summarize_execution_result(user_message, plan, execution_result, store)
    return StreamingResponse(instant_stream_with_nodes(local_summary, referenced_nodes), media_type="text/event-stream")
