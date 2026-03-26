"""
ingest.py — SAP Order-to-Cash Graph Builder

Reads JSONL files from the sap-o2c-data directory, cleans and normalises values,
constructs a NetworkX graph with typed nodes and edges, flags orphaned foreign
keys, and serialises the result to graph.json.

Usage:
    python ingest.py <path-to-sap-o2c-data>
    python ingest.py /Users/harshbatra/Downloads/sap-o2c-data
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# 1.  JSONL Loading
# ---------------------------------------------------------------------------


def load_jsonl_folder(folder: Path) -> list[dict[str, Any]]:
    """Read all .jsonl part-files in *folder* and return a flat list of dicts."""
    records: list[dict[str, Any]] = []
    for part in sorted(folder.glob("*.jsonl")):
        with part.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_all_entities(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load every entity folder under *data_dir* into a name → records map."""
    entities: dict[str, list[dict[str, Any]]] = {}
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            entities[child.name] = load_jsonl_folder(child)
    return entities


# ---------------------------------------------------------------------------
# 2.  Data Cleaning / Normalisation
# ---------------------------------------------------------------------------

# Fields we know are numeric (float) — values arrive as strings like "533.05"
FLOAT_FIELDS: set[str] = {
    "totalNetAmount", "netAmount", "billingQuantity", "requestedQuantity",
    "actualDeliveryQuantity", "grossWeight", "netWeight",
    "amountInTransactionCurrency", "amountInCompanyCodeCurrency",
    "confdOrderQtyByMatlAvailCheck",
}

# Fields we know are integer (or integer-like identifiers stored as strings)
# We intentionally keep document IDs as strings to preserve leading zeros.


def _clean_time_obj(val: dict[str, int]) -> str:
    """Convert {"hours": 11, "minutes": 31, "seconds": 13} → "11:31:13"."""
    return f"{val['hours']:02d}:{val['minutes']:02d}:{val['seconds']:02d}"


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply cleaning rules to a single record in-place and return it.

    Rules:
    - Empty strings → None
    - Time objects → HH:MM:SS strings
    - Numeric-string fields → float
    - Date strings are kept as-is (ISO format already)
    """
    cleaned: dict[str, Any] = {}
    for key, val in record.items():
        # Empty string → None
        if val == "":
            cleaned[key] = None
            continue

        # Time objects (SAP-style {hours, minutes, seconds})
        if isinstance(val, dict) and set(val.keys()) == {"hours", "minutes", "seconds"}:
            cleaned[key] = _clean_time_obj(val)
            continue

        # Numeric strings
        if key in FLOAT_FIELDS and isinstance(val, str):
            try:
                cleaned[key] = float(val)
            except ValueError:
                cleaned[key] = val
            continue

        # Everything else passes through unchanged
        cleaned[key] = val

    return cleaned


def clean_all(entities: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Apply cleaning to every record in every entity table."""
    return {
        name: [clean_record(r) for r in records]
        for name, records in entities.items()
    }


# ---------------------------------------------------------------------------
# 3.  Lookup Index Builders
# ---------------------------------------------------------------------------


def build_index(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Build a dict mapping key-field values → record (last-write-wins)."""
    return {r[key]: r for r in records if r.get(key) is not None}


def build_multi_index(
    records: list[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    """Build a dict mapping key-field values → list of records."""
    idx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        k = r.get(key)
        if k is not None:
            idx[k].append(r)
    return dict(idx)


# ---------------------------------------------------------------------------
# 4.  Graph Construction
# ---------------------------------------------------------------------------

# Warnings log for orphaned references
warnings: list[str] = []


def _node_id(node_type: str, *parts: str) -> str:
    """Canonical node ID:  'BillingDocument:90504298'."""
    return f"{node_type}:{':'.join(parts)}"


def _add_node(
    G: nx.DiGraph,
    node_type: str,
    key_parts: tuple[str, ...],
    props: dict[str, Any],
    *,
    incomplete: bool = False,
) -> str:
    """Add a node (or update it in-place) and return its ID."""
    nid = _node_id(node_type, *key_parts)
    attrs = {"node_type": node_type, **props}
    if incomplete:
        attrs["incomplete"] = True
    if nid in G:
        # Merge: existing attrs stay, new attrs added/overwritten
        G.nodes[nid].update(attrs)
    else:
        G.add_node(nid, **attrs)
    return nid


def _ensure_node(G: nx.DiGraph, node_type: str, key: str) -> str:
    """Return the node ID, creating an incomplete stub if it doesn't exist."""
    nid = _node_id(node_type, key)
    if nid not in G:
        warnings.append(f"Orphan: {node_type} '{key}' referenced but not in dataset — created as incomplete")
        G.add_node(nid, node_type=node_type, incomplete=True)
    return nid


def _add_edge(
    G: nx.DiGraph,
    src: str,
    tgt: str,
    edge_type: str,
    props: dict[str, Any] | None = None,
) -> None:
    """Add a directed edge with an edge_type label."""
    G.add_edge(src, tgt, edge_type=edge_type, **(props or {}))


def build_graph(entities: dict[str, list[dict[str, Any]]]) -> nx.DiGraph:
    """Construct the full O2C graph from cleaned entity tables."""
    G = nx.DiGraph()

    # ---------------------------------------------------------------
    # a) Customers  (merge business_partners + addresses + assignments)
    # ---------------------------------------------------------------
    bp_index = build_index(entities.get("business_partners", []), "businessPartner")
    addr_index = build_index(entities.get("business_partner_addresses", []), "businessPartner")
    cca_index = build_index(entities.get("customer_company_assignments", []), "customer")
    csa_multi = build_multi_index(entities.get("customer_sales_area_assignments", []), "customer")

    # Collect all customer IDs from any of the four tables
    all_customer_ids: set[str] = set()
    for r in entities.get("business_partners", []):
        all_customer_ids.add(r.get("customer") or r["businessPartner"])
    for r in entities.get("business_partner_addresses", []):
        all_customer_ids.add(r["businessPartner"])
    for r in entities.get("customer_company_assignments", []):
        all_customer_ids.add(r["customer"])
    for r in entities.get("customer_sales_area_assignments", []):
        all_customer_ids.add(r["customer"])

    for cid in all_customer_ids:
        props: dict[str, Any] = {"customer_id": cid}
        bp = bp_index.get(cid, {})
        addr = addr_index.get(cid, {})
        cca = cca_index.get(cid, {})
        csas = csa_multi.get(cid, [])

        # From business_partners
        props["name"] = bp.get("businessPartnerFullName")
        props["category"] = bp.get("businessPartnerCategory")
        props["grouping"] = bp.get("businessPartnerGrouping")
        props["isBlocked"] = bp.get("businessPartnerIsBlocked")
        props["createdBy"] = bp.get("createdByUser")
        props["creationDate"] = bp.get("creationDate")

        # From address
        props["city"] = addr.get("cityName")
        props["country"] = addr.get("country")
        props["region"] = addr.get("region")
        props["postalCode"] = addr.get("postalCode")
        props["streetName"] = addr.get("streetName")

        # From company assignment
        props["companyCode"] = cca.get("companyCode")
        props["reconciliationAccount"] = cca.get("reconciliationAccount")
        props["accountGroup"] = cca.get("customerAccountGroup")
        props["paymentTerms"] = cca.get("paymentTerms")

        # From sales area assignments (can be multiple per customer)
        if csas:
            props["salesAreaAssignments"] = [
                {
                    "salesOrganization": sa.get("salesOrganization"),
                    "distributionChannel": sa.get("distributionChannel"),
                    "division": sa.get("division"),
                    "currency": sa.get("currency"),
                    "incoterms": sa.get("incotermsClassification"),
                    "paymentTerms": sa.get("customerPaymentTerms"),
                    "shippingCondition": sa.get("shippingCondition"),
                }
                for sa in csas
            ]

        _add_node(G, "Customer", (cid,), props)

    # ---------------------------------------------------------------
    # b) Products  (merge products + product_descriptions)
    # ---------------------------------------------------------------
    desc_index = build_index(entities.get("product_descriptions", []), "product")

    for rec in entities.get("products", []):
        pid = rec["product"]
        desc = desc_index.get(pid, {})
        props = {
            "product_id": pid,
            "description": desc.get("productDescription"),
            "productType": rec.get("productType"),
            "productGroup": rec.get("productGroup"),
            "grossWeight": rec.get("grossWeight"),
            "netWeight": rec.get("netWeight"),
            "weightUnit": rec.get("weightUnit"),
            "baseUnit": rec.get("baseUnit"),
            "division": rec.get("division"),
            "industrySector": rec.get("industrySector"),
            "creationDate": rec.get("creationDate"),
            "isMarkedForDeletion": rec.get("isMarkedForDeletion"),
        }
        _add_node(G, "Product", (pid,), props)

    # ---------------------------------------------------------------
    # c) Plants
    # ---------------------------------------------------------------
    for rec in entities.get("plants", []):
        pid = rec["plant"]
        props = {
            "plant_id": pid,
            "plantName": rec.get("plantName"),
            "salesOrganization": rec.get("salesOrganization"),
            "distributionChannel": rec.get("distributionChannel"),
            "division": rec.get("division"),
            "language": rec.get("language"),
            "factoryCalendar": rec.get("factoryCalendar"),
        }
        _add_node(G, "Plant", (pid,), props)

    # Product ↔ Plant edges (from product_plants, aggregated)
    for rec in entities.get("product_plants", []):
        prod_nid = _ensure_node(G, "Product", rec["product"])
        plant_nid = _ensure_node(G, "Plant", rec["plant"])
        _add_edge(G, prod_nid, plant_nid, "AVAILABLE_AT", {
            "profitCenter": rec.get("profitCenter"),
            "mrpType": rec.get("mrpType"),
        })

    # ---------------------------------------------------------------
    # d) Sales Orders (headers + items + schedule lines)
    # ---------------------------------------------------------------
    sched_multi = build_multi_index(
        entities.get("sales_order_schedule_lines", []),
        "salesOrder",
    )
    # Sub-index schedule lines by (salesOrder, salesOrderItem)
    sched_by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in entities.get("sales_order_schedule_lines", []):
        sched_by_item[(s["salesOrder"], s["salesOrderItem"])].append(s)

    for rec in entities.get("sales_order_headers", []):
        so_id = rec["salesOrder"]
        props = {
            "salesOrder_id": so_id,
            "salesOrderType": rec.get("salesOrderType"),
            "salesOrganization": rec.get("salesOrganization"),
            "distributionChannel": rec.get("distributionChannel"),
            "totalNetAmount": rec.get("totalNetAmount"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "creationDate": rec.get("creationDate"),
            "overallDeliveryStatus": rec.get("overallDeliveryStatus"),
            "requestedDeliveryDate": rec.get("requestedDeliveryDate"),
            "incoterms": rec.get("incotermsClassification"),
            "paymentTerms": rec.get("customerPaymentTerms"),
        }
        so_nid = _add_node(G, "SalesOrder", (so_id,), props)

        # Edge → Customer (soldToParty)
        cust_id = rec.get("soldToParty")
        if cust_id:
            cust_nid = _ensure_node(G, "Customer", cust_id)
            _add_edge(G, so_nid, cust_nid, "SOLD_TO")

    for rec in entities.get("sales_order_items", []):
        so_id = rec["salesOrder"]
        item_id = rec["salesOrderItem"]
        composite = f"{so_id}:{item_id}"

        # Merge schedule line data (confirmedDeliveryDate)
        sched_lines = sched_by_item.get((so_id, item_id), [])
        confirmed_date = sched_lines[0].get("confirmedDeliveryDate") if sched_lines else None

        props = {
            "salesOrder_id": so_id,
            "salesOrderItem_id": item_id,
            "material": rec.get("material"),
            "requestedQuantity": rec.get("requestedQuantity"),
            "requestedQuantityUnit": rec.get("requestedQuantityUnit"),
            "netAmount": rec.get("netAmount"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "materialGroup": rec.get("materialGroup"),
            "productionPlant": rec.get("productionPlant"),
            "confirmedDeliveryDate": confirmed_date,
        }
        soi_nid = _add_node(G, "SalesOrderItem", (so_id, item_id), props)

        # Edge → parent SalesOrder
        so_nid = _ensure_node(G, "SalesOrder", so_id)
        _add_edge(G, so_nid, soi_nid, "HAS_ITEM")

        # Edge → Product
        mat = rec.get("material")
        if mat:
            prod_nid = _ensure_node(G, "Product", mat)
            _add_edge(G, soi_nid, prod_nid, "CONTAINS_PRODUCT")

    # ---------------------------------------------------------------
    # e) Outbound Deliveries (headers + items)
    # ---------------------------------------------------------------
    for rec in entities.get("outbound_delivery_headers", []):
        dd_id = rec["deliveryDocument"]
        props = {
            "deliveryDocument_id": dd_id,
            "creationDate": rec.get("creationDate"),
            "goodsMovementDate": rec.get("actualGoodsMovementDate"),
            "goodsMovementStatus": rec.get("overallGoodsMovementStatus"),
            "pickingStatus": rec.get("overallPickingStatus"),
            "shippingPoint": rec.get("shippingPoint"),
        }
        _add_node(G, "DeliveryDocument", (dd_id,), props)

    for rec in entities.get("outbound_delivery_items", []):
        dd_id = rec["deliveryDocument"]
        item_id = rec["deliveryDocumentItem"]
        props = {
            "deliveryDocument_id": dd_id,
            "deliveryDocumentItem_id": item_id,
            "actualDeliveryQuantity": rec.get("actualDeliveryQuantity"),
            "deliveryQuantityUnit": rec.get("deliveryQuantityUnit"),
            "plant": rec.get("plant"),
            "storageLocation": rec.get("storageLocation"),
        }
        di_nid = _add_node(G, "DeliveryItem", (dd_id, item_id), props)

        # Edge → parent DeliveryDocument
        dd_nid = _ensure_node(G, "DeliveryDocument", dd_id)
        _add_edge(G, dd_nid, di_nid, "HAS_ITEM")

        # Edge → SalesOrder (referenceSdDocument = salesOrder ID)
        ref_so = rec.get("referenceSdDocument")
        if ref_so:
            so_nid = _ensure_node(G, "SalesOrder", ref_so)
            _add_edge(G, dd_nid, so_nid, "FULFILLS_ORDER")

        # Edge → Plant
        plant_id = rec.get("plant")
        if plant_id:
            plant_nid = _ensure_node(G, "Plant", plant_id)
            _add_edge(G, di_nid, plant_nid, "SHIPS_FROM")

    # ---------------------------------------------------------------
    # f) Billing Documents (headers + items + cancellations)
    # ---------------------------------------------------------------
    for rec in entities.get("billing_document_headers", []):
        bd_id = rec["billingDocument"]
        props = {
            "billingDocument_id": bd_id,
            "billingDocumentType": rec.get("billingDocumentType"),
            "billingDocumentDate": rec.get("billingDocumentDate"),
            "totalNetAmount": rec.get("totalNetAmount"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "isCancelled": rec.get("billingDocumentIsCancelled"),
            "companyCode": rec.get("companyCode"),
            "fiscalYear": rec.get("fiscalYear"),
            "accountingDocument": rec.get("accountingDocument"),
            "creationDate": rec.get("creationDate"),
        }
        bd_nid = _add_node(G, "BillingDocument", (bd_id,), props)

        # Edge → Customer (soldToParty)
        cust_id = rec.get("soldToParty")
        if cust_id:
            cust_nid = _ensure_node(G, "Customer", cust_id)
            _add_edge(G, bd_nid, cust_nid, "BILLED_TO")

    for rec in entities.get("billing_document_items", []):
        bd_id = rec["billingDocument"]
        item_id = rec["billingDocumentItem"]
        props = {
            "billingDocument_id": bd_id,
            "billingDocumentItem_id": item_id,
            "material": rec.get("material"),
            "billingQuantity": rec.get("billingQuantity"),
            "billingQuantityUnit": rec.get("billingQuantityUnit"),
            "netAmount": rec.get("netAmount"),
            "transactionCurrency": rec.get("transactionCurrency"),
        }
        bi_nid = _add_node(G, "BillingItem", (bd_id, item_id), props)

        # Edge → parent BillingDocument
        bd_nid = _ensure_node(G, "BillingDocument", bd_id)
        _add_edge(G, bd_nid, bi_nid, "HAS_ITEM")

        # Edge → DeliveryDocument (referenceSdDocument = delivery doc ID)
        ref_dd = rec.get("referenceSdDocument")
        if ref_dd:
            dd_nid = _ensure_node(G, "DeliveryDocument", ref_dd)
            _add_edge(G, bd_nid, dd_nid, "BILLS_DELIVERY")

        # Edge → Product
        mat = rec.get("material")
        if mat:
            prod_nid = _ensure_node(G, "Product", mat)
            _add_edge(G, bi_nid, prod_nid, "BILLS_PRODUCT")

    # Cancellations  (billing_document_cancellations are billing docs that cancel others)
    for rec in entities.get("billing_document_cancellations", []):
        cancelling_id = rec["billingDocument"]
        cancelled_id = rec.get("cancelledBillingDocument")

        # Also add/update the cancelling doc as a BillingDocument node
        props = {
            "billingDocument_id": cancelling_id,
            "billingDocumentType": rec.get("billingDocumentType"),
            "billingDocumentDate": rec.get("billingDocumentDate"),
            "totalNetAmount": rec.get("totalNetAmount"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "isCancelled": rec.get("billingDocumentIsCancelled"),
            "companyCode": rec.get("companyCode"),
            "fiscalYear": rec.get("fiscalYear"),
            "accountingDocument": rec.get("accountingDocument"),
            "creationDate": rec.get("creationDate"),
            "isCancellation": True,  # flag this doc as a cancellation doc
        }
        canc_nid = _add_node(G, "BillingDocument", (cancelling_id,), props)

        # Edge → cancelled billing doc (if the reference exists and is non-empty)
        if cancelled_id:
            orig_nid = _ensure_node(G, "BillingDocument", cancelled_id)
            _add_edge(G, canc_nid, orig_nid, "CANCELLED_BY")

        # Edge → Customer
        cust_id = rec.get("soldToParty")
        if cust_id:
            cust_nid = _ensure_node(G, "Customer", cust_id)
            _add_edge(G, canc_nid, cust_nid, "BILLED_TO")

    # ---------------------------------------------------------------
    # g) Journal Entries (accounts receivable)
    # ---------------------------------------------------------------
    for rec in entities.get("journal_entry_items_accounts_receivable", []):
        acct_doc = rec["accountingDocument"]
        item_id = rec["accountingDocumentItem"]
        props = {
            "accountingDocument_id": acct_doc,
            "accountingDocumentItem_id": item_id,
            "glAccount": rec.get("glAccount"),
            "amountInTransactionCurrency": rec.get("amountInTransactionCurrency"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "amountInCompanyCodeCurrency": rec.get("amountInCompanyCodeCurrency"),
            "companyCodeCurrency": rec.get("companyCodeCurrency"),
            "postingDate": rec.get("postingDate"),
            "documentDate": rec.get("documentDate"),
            "accountingDocumentType": rec.get("accountingDocumentType"),
            "clearingDate": rec.get("clearingDate"),
            "clearingAccountingDocument": rec.get("clearingAccountingDocument"),
            "profitCenter": rec.get("profitCenter"),
            "companyCode": rec.get("companyCode"),
            "fiscalYear": rec.get("fiscalYear"),
        }
        je_nid = _add_node(G, "JournalEntry", (acct_doc, str(item_id)), props)

        # Edge → BillingDocument (referenceDocument = billing doc ID)
        ref_bd = rec.get("referenceDocument")
        if ref_bd:
            bd_nid = _ensure_node(G, "BillingDocument", ref_bd)
            _add_edge(G, je_nid, bd_nid, "RECORDS_BILLING")

        # Edge → Customer
        cust_id = rec.get("customer")
        if cust_id:
            cust_nid = _ensure_node(G, "Customer", cust_id)
            _add_edge(G, je_nid, cust_nid, "FOR_CUSTOMER")

    # ---------------------------------------------------------------
    # h) Payments (accounts receivable)
    # ---------------------------------------------------------------
    for rec in entities.get("payments_accounts_receivable", []):
        acct_doc = rec["accountingDocument"]
        item_id = rec["accountingDocumentItem"]
        props = {
            "accountingDocument_id": acct_doc,
            "accountingDocumentItem_id": item_id,
            "amountInTransactionCurrency": rec.get("amountInTransactionCurrency"),
            "transactionCurrency": rec.get("transactionCurrency"),
            "amountInCompanyCodeCurrency": rec.get("amountInCompanyCodeCurrency"),
            "companyCodeCurrency": rec.get("companyCodeCurrency"),
            "postingDate": rec.get("postingDate"),
            "documentDate": rec.get("documentDate"),
            "clearingDate": rec.get("clearingDate"),
            "clearingAccountingDocument": rec.get("clearingAccountingDocument"),
            "glAccount": rec.get("glAccount"),
            "profitCenter": rec.get("profitCenter"),
            "companyCode": rec.get("companyCode"),
            "fiscalYear": rec.get("fiscalYear"),
            "invoiceReference": rec.get("invoiceReference"),
        }
        pay_nid = _add_node(G, "Payment", (acct_doc, str(item_id)), props)

        # Edge → JournalEntry (via clearingAccountingDocument)
        clearing_doc = rec.get("clearingAccountingDocument")
        if clearing_doc:
            # Payments clear journal entries — match on the clearing doc's accounting document
            # We link to the first item of that journal entry
            je_nid = _ensure_node(G, "JournalEntry", clearing_doc)
            _add_edge(G, pay_nid, je_nid, "CLEARS_JOURNAL")

        # Edge → Customer
        cust_id = rec.get("customer")
        if cust_id:
            cust_nid = _ensure_node(G, "Customer", cust_id)
            _add_edge(G, pay_nid, cust_nid, "PAID_BY")

    return G


# ---------------------------------------------------------------------------
# 5.  Serialisation
# ---------------------------------------------------------------------------


def graph_to_serialisable(G: nx.DiGraph) -> dict[str, Any]:
    """Convert a NetworkX graph into a JSON-safe dict.

    Non-serialisable values (sets, etc.) are converted to strings.
    """
    nodes = []
    for nid, attrs in G.nodes(data=True):
        node_data = {"id": nid, **attrs}
        # Convert any non-serialisable values
        for k, v in node_data.items():
            if isinstance(v, (set, frozenset)):
                node_data[k] = list(v)
        nodes.append(node_data)

    edges = []
    for src, tgt, attrs in G.edges(data=True):
        edge_data = {"source": src, "target": tgt, **attrs}
        edges.append(edge_data)

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# 6.  Summary / Reporting
# ---------------------------------------------------------------------------


def print_summary(G: nx.DiGraph) -> None:
    """Print node and edge counts grouped by type, plus any warnings."""
    print("\n" + "=" * 60)
    print("  GRAPH CONSTRUCTION SUMMARY")
    print("=" * 60)

    # Node counts by type
    node_types: dict[str, int] = defaultdict(int)
    incomplete_count = 0
    for _, attrs in G.nodes(data=True):
        node_types[attrs.get("node_type", "UNKNOWN")] += 1
        if attrs.get("incomplete"):
            incomplete_count += 1

    print(f"\n  Total Nodes: {G.number_of_nodes()}")
    print(f"  Total Edges: {G.number_of_edges()}")
    print(f"  Incomplete (orphan) Nodes: {incomplete_count}")

    print("\n  Nodes by Type:")
    for ntype in sorted(node_types):
        print(f"    {ntype:25s}  {node_types[ntype]:>6d}")

    # Edge counts by type
    edge_types: dict[str, int] = defaultdict(int)
    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get("edge_type", "UNKNOWN")] += 1

    print("\n  Edges by Type:")
    for etype in sorted(edge_types):
        print(f"    {etype:25s}  {edge_types[etype]:>6d}")

    # Warnings
    if warnings:
        print(f"\n  ⚠  Warnings ({len(warnings)}):")
        # Show deduplicated summary
        from collections import Counter
        warning_counts = Counter(warnings)
        for msg, count in warning_counts.most_common(20):
            suffix = f" (×{count})" if count > 1 else ""
            print(f"    - {msg}{suffix}")
        if len(warning_counts) > 20:
            print(f"    ... and {len(warning_counts) - 20} more unique warnings")
    else:
        print("\n  ✓  No warnings — all references resolved.")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-sap-o2c-data>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' is not a directory")
        sys.exit(1)

    output_path = Path(__file__).parent / "graph.json"

    # Step 1: Load
    print(f"Loading entities from {data_dir} ...")
    entities = load_all_entities(data_dir)
    total_records = sum(len(recs) for recs in entities.values())
    print(f"  Loaded {len(entities)} entity tables, {total_records:,} total records")

    # Step 2: Clean
    print("Cleaning & normalising ...")
    entities = clean_all(entities)

    # Step 3: Build graph
    print("Building graph ...")
    G = build_graph(entities)

    # Step 4: Summarise
    print_summary(G)

    # Step 5: Serialise
    print(f"\nSerialising graph to {output_path} ...")
    serialisable = graph_to_serialisable(G)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, default=str)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Written {output_path} ({file_size_mb:.1f} MB)")
    print("  Done ✓")


if __name__ == "__main__":
    main()
