"""
verify_graph.py — Manual verification of the ingested O2C graph

Loads graph.json and runs two test queries:
1. All neighbours of a known SalesOrder node
2. Full O2C chain for a known BillingDocument

Usage:
    python verify_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx


def load_graph(path: Path) -> nx.DiGraph:
    """Reconstruct a NetworkX DiGraph from the serialised graph.json."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    G = nx.DiGraph()

    for node in data["nodes"]:
        nid = node.pop("id")
        G.add_node(nid, **node)

    for edge in data["edges"]:
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)

    return G


def fmt_node(G: nx.DiGraph, nid: str, indent: int = 4) -> str:
    """Format a node for display: type, ID, and key properties."""
    attrs = G.nodes[nid]
    node_type = attrs.get("node_type", "?")
    # Pick a human-readable label
    label_keys = ["name", "description", "plantName", "billingDocument_id",
                  "salesOrder_id", "deliveryDocument_id", "accountingDocument_id",
                  "customer_id", "product_id", "plant_id"]
    label = ""
    for k in label_keys:
        if attrs.get(k):
            label = f' "{attrs[k]}"'
            break

    incomplete = " [INCOMPLETE]" if attrs.get("incomplete") else ""
    prefix = " " * indent
    return f"{prefix}{node_type}: {nid}{label}{incomplete}"


def print_neighbours(G: nx.DiGraph, node_id: str) -> None:
    """Print all direct neighbours (in + out) of a node with edge types."""
    if node_id not in G:
        print(f"  ✗ Node '{node_id}' not found in graph")
        return

    print(f"\n  Node: {fmt_node(G, node_id, indent=0)}")
    print(f"  {'-' * 50}")

    # Outgoing edges
    out_edges = list(G.out_edges(node_id, data=True))
    if out_edges:
        print(f"\n  Outgoing edges ({len(out_edges)}):")
        for _, tgt, attrs in out_edges:
            etype = attrs.get("edge_type", "?")
            print(f"    --[{etype}]--> {fmt_node(G, tgt, indent=0)}")

    # Incoming edges
    in_edges = list(G.in_edges(node_id, data=True))
    if in_edges:
        print(f"\n  Incoming edges ({len(in_edges)}):")
        for src, _, attrs in in_edges:
            etype = attrs.get("edge_type", "?")
            print(f"    <--[{etype}]-- {fmt_node(G, src, indent=0)}")

    if not out_edges and not in_edges:
        print("  (isolated node — no edges)")


def trace_o2c_chain(G: nx.DiGraph, billing_doc_id: str) -> None:
    """Trace the full Order-to-Cash chain starting from a BillingDocument.

    BillingDocument → (BILLS_DELIVERY) → DeliveryDocument → (FULFILLS_ORDER) → SalesOrder → (SOLD_TO) → Customer
    BillingDocument → (BILLED_TO) → Customer
    JournalEntry → (RECORDS_BILLING) → BillingDocument  (incoming)
    Payment → (CLEARS_JOURNAL) → JournalEntry           (incoming)
    """
    bd_nid = f"BillingDocument:{billing_doc_id}"
    if bd_nid not in G:
        print(f"  ✗ BillingDocument '{billing_doc_id}' not found in graph")
        return

    print(f"\n  {'=' * 60}")
    print(f"  FULL O2C CHAIN for BillingDocument {billing_doc_id}")
    print(f"  {'=' * 60}")

    visited: set[str] = set()

    def _walk(nid: str, depth: int = 0) -> None:
        if nid in visited:
            return
        visited.add(nid)

        indent = "  " + "  │ " * depth
        attrs = G.nodes[nid]
        node_type = attrs.get("node_type", "?")

        # Human-readable summary based on type
        summary_parts: list[str] = []
        if node_type == "BillingDocument":
            amt = attrs.get("totalNetAmount")
            ccy = attrs.get("transactionCurrency", "")
            cancelled = " [CANCELLED]" if attrs.get("isCancelled") else ""
            summary_parts.append(f"Amount: {amt} {ccy}{cancelled}")
            summary_parts.append(f"Date: {attrs.get('billingDocumentDate', '?')}")
        elif node_type == "DeliveryDocument":
            summary_parts.append(f"Status: {attrs.get('goodsMovementStatus', '?')}")
            summary_parts.append(f"Ship Point: {attrs.get('shippingPoint', '?')}")
        elif node_type == "SalesOrder":
            summary_parts.append(f"Amount: {attrs.get('totalNetAmount')} {attrs.get('transactionCurrency', '')}")
            summary_parts.append(f"Delivery Status: {attrs.get('overallDeliveryStatus', '?')}")
        elif node_type == "Customer":
            summary_parts.append(f"Name: {attrs.get('name', '?')}")
            summary_parts.append(f"City: {attrs.get('city', '?')}")
        elif node_type == "JournalEntry":
            summary_parts.append(f"Amount: {attrs.get('amountInTransactionCurrency')} {attrs.get('transactionCurrency', '')}")
            summary_parts.append(f"GL: {attrs.get('glAccount', '?')}")
        elif node_type == "Payment":
            summary_parts.append(f"Amount: {attrs.get('amountInTransactionCurrency')} {attrs.get('transactionCurrency', '')}")
            summary_parts.append(f"Clearing Doc: {attrs.get('clearingAccountingDocument', '?')}")

        summary = " | ".join(summary_parts) if summary_parts else ""
        incomplete = " [INCOMPLETE]" if attrs.get("incomplete") else ""
        print(f"{indent}📦 {node_type}: {nid}{incomplete}")
        if summary:
            print(f"{indent}   {summary}")

        # Walk outgoing edges in O2C order
        for _, tgt, eattrs in G.out_edges(nid, data=True):
            etype = eattrs.get("edge_type", "?")
            tgt_type = G.nodes[tgt].get("node_type", "?")
            # Only follow O2C-relevant edges (not HAS_ITEM etc for brevity in chain view)
            if etype in ("BILLS_DELIVERY", "FULFILLS_ORDER", "SOLD_TO", "BILLED_TO",
                         "CANCELLED_BY"):
                print(f"{indent}   └──[{etype}]──▸")
                _walk(tgt, depth + 1)

        # Walk incoming edges (journal entries and payments referencing this node)
        for src, _, eattrs in G.in_edges(nid, data=True):
            etype = eattrs.get("edge_type", "?")
            if etype in ("RECORDS_BILLING", "CLEARS_JOURNAL"):
                print(f"{indent}   ◂──[{etype}]──┘")
                _walk(src, depth + 1)

    _walk(bd_nid)

    # Also show billing items
    print(f"\n  Line Items:")
    for _, tgt, eattrs in G.out_edges(bd_nid, data=True):
        if eattrs.get("edge_type") == "HAS_ITEM":
            item_attrs = G.nodes[tgt]
            mat = item_attrs.get("material", "?")
            qty = item_attrs.get("billingQuantity", "?")
            amt = item_attrs.get("netAmount", "?")
            ccy = item_attrs.get("transactionCurrency", "")
            print(f"    • Item {tgt}: material={mat}, qty={qty}, amount={amt} {ccy}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    graph_path = Path(__file__).parent / "graph.json"
    if not graph_path.exists():
        print(f"Error: {graph_path} not found. Run ingest.py first.")
        sys.exit(1)

    print(f"Loading graph from {graph_path} ...")
    G = load_graph(graph_path)
    print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # ---------------------------------------------------------------
    # Test 1:  All neighbours of a known SalesOrder
    # ---------------------------------------------------------------
    # Pick the first SalesOrder we can find
    sales_orders = [n for n, d in G.nodes(data=True) if d.get("node_type") == "SalesOrder"
                    and not d.get("incomplete")]
    if sales_orders:
        test_so = sales_orders[0]
        print("=" * 60)
        print(f"TEST 1: Neighbours of {test_so}")
        print("=" * 60)
        print_neighbours(G, test_so)
    else:
        print("  ✗ No SalesOrder nodes found!")

    print("\n")

    # ---------------------------------------------------------------
    # Test 2:  Full O2C chain for a known BillingDocument
    # ---------------------------------------------------------------
    billing_docs = [n for n, d in G.nodes(data=True)
                    if d.get("node_type") == "BillingDocument" and not d.get("incomplete")]
    if billing_docs:
        # Pick one that has outgoing BILLS_DELIVERY edges for a richer trace
        test_bd = None
        for bd in billing_docs:
            out_types = [d.get("edge_type") for _, _, d in G.out_edges(bd, data=True)]
            if "BILLS_DELIVERY" in out_types:
                test_bd = bd
                break
        if test_bd is None:
            test_bd = billing_docs[0]

        bd_id = test_bd.split(":", 1)[1]
        print("=" * 60)
        print(f"TEST 2: Full O2C chain for BillingDocument {bd_id}")
        print("=" * 60)
        trace_o2c_chain(G, bd_id)
    else:
        print("  ✗ No BillingDocument nodes found!")


if __name__ == "__main__":
    main()
