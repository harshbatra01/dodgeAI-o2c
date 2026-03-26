"""
graph_store.py — NetworkX Graph State Management

Loads the serialized graph.json into memory and provides query methods
for the API endpoints.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

class GraphStore:
    def __init__(self, filepath: Path | str):
        """Initialise the GraphStore by loading the graph from disk."""
        self.filepath = Path(filepath)
        self.G = self._load_graph(self.filepath)
        self._undirected_graph = self.G.to_undirected()
        self._lexical_tokens: set[str] = set()
        self._lexical_tokens_by_type: dict[str, set[str]] = defaultdict(set)
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._semantic_node_ids: list[str] = []
        self._semantic_embeddings = np.empty((0, 0), dtype=np.float32)
        self._graph_analysis: dict[str, Any] = {}
        self._build_lexical_index()
        self._build_semantic_index()
        self._build_graph_analysis()

    def _load_graph(self, path: Path) -> nx.DiGraph:
        """Read the JSON-serialised DiGraph and reconstruct it."""
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found at {path}")

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        G = nx.DiGraph()

        for node_data in data.get("nodes", []):
            nid = node_data.pop("id")
            G.add_node(nid, **node_data)

        for edge_data in data.get("edges", []):
            src = edge_data.pop("source")
            tgt = edge_data.pop("target")
            G.add_edge(src, tgt, **edge_data)

        return G

    def get_stats(self) -> dict[str, Any]:
        """Compute basic summary statistics of the graph."""
        node_counts: dict[str, int] = {}
        for _, attrs in self.G.nodes(data=True):
            ntype = attrs.get("node_type", "UNKNOWN")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, attrs in self.G.edges(data=True):
            etype = attrs.get("edge_type", "UNKNOWN")
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.G

    def get_node_with_neighbours(self, node_id: str) -> dict[str, Any]:
        """Return a node\'s properties and all its direct contiguous neighbours."""
        if node_id not in self.G:
            return {}

        node_props = dict(self.G.nodes[node_id])
        node_props["id"] = node_id

        # Collect edges connected to the central node
        edges = []
        neighbour_ids = set()

        for u, v, attrs in self.G.out_edges(node_id, data=True):
            edges.append({"source": u, "target": v, **attrs})
            neighbour_ids.add(v)

        for u, v, attrs in self.G.in_edges(node_id, data=True):
            edges.append({"source": u, "target": v, **attrs})
            neighbour_ids.add(u)

        # Collect neighbour node properties
        nodes = [node_props]
        for nid in neighbour_ids:
            props = dict(self.G.nodes[nid])
            props["id"] = nid
            nodes.append(props)

        return {
            "nodes": nodes,
            "edges": edges
        }

    def get_ego_graph(self, center_id: str, radius: int) -> dict[str, Any]:
        """Return the subgraph within *radius* hops surrounding *center_id*."""
        if center_id not in self.G:
            return {"nodes": [], "edges": []}

        # Ego graph gets all nodes within distance 'radius' from center
        sub_g = nx.ego_graph(self.G, center_id, radius=radius, undirected=True)

        nodes = []
        for nid, attrs in sub_g.nodes(data=True):
            nodes.append({"id": nid, **attrs})

        edges = []
        for u, v, attrs in sub_g.edges(data=True):
            edges.append({"source": u, "target": v, **attrs})

        return {
            "nodes": nodes,
            "edges": edges
        }

    def search_nodes(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find nodes matching the search substring (case-insensitive)."""
        query_lower = query.lower()
        results = []

        # Iterate all nodes and stringify their attributes for matching
        for nid, attrs in self.G.nodes(data=True):
            # Form an ad-hoc search string containing the ID and key fields
            search_content = f"{nid} " + " ".join(
                str(v) for k, v in attrs.items() 
                if isinstance(v, (str, int, float))
            )
            if query_lower in search_content.lower():
                results.append({"id": nid, **attrs})
                if len(results) >= limit:
                    break
        
        return results

    def semantic_search_nodes(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find nodes whose embedding is most similar to the query embedding."""
        if not query.strip() or self._semantic_embeddings.size == 0:
            return []

        query_embedding = self._embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._semantic_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]
        per_type_limit = max(1, min(4, limit))
        type_counts: dict[str, int] = {}
        top_indices: list[int] = []

        for idx in ranked_indices:
            node_id = self._semantic_node_ids[idx]
            node_type = self.G.nodes[node_id].get("node_type", node_id.split(":", 1)[0])
            if type_counts.get(node_type, 0) >= per_type_limit:
                continue
            top_indices.append(int(idx))
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            if len(top_indices) >= limit:
                break

        results = []
        for idx in top_indices:
            node_id = self._semantic_node_ids[idx]
            attrs = dict(self.G.nodes[node_id])
            results.append({
                "id": node_id,
                "similarity": float(scores[idx]),
                **attrs,
            })
        return results

    def filter_query_tokens(self, query: str, node_type: str | None = None) -> list[str]:
        """Return tokens that exist in the graph lexicon (optionally type-scoped)."""
        tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
        if not tokens:
            return []
        lexicon = self._lexical_tokens_by_type.get(node_type) if node_type else self._lexical_tokens
        if not lexicon:
            return tokens
        return [token for token in tokens if token in lexicon]

    def get_graph_analysis_overview(self) -> dict[str, Any]:
        """Return cluster and centrality metrics for the full graph."""
        return self._graph_analysis["overview"]

    def get_graph_clusters(self, limit: int = 10) -> dict[str, Any]:
        """Return summaries for the largest graph communities."""
        clusters = self._graph_analysis["clusters"][:limit]
        return {
            "clusters": clusters,
            "total_clusters": self._graph_analysis["overview"]["total_clusters"],
        }

    def get_cluster_detail(self, cluster_id: int) -> dict[str, Any]:
        """Return the full node list and metadata for a specific cluster."""
        for cluster in self._graph_analysis["clusters"]:
            if cluster["cluster_id"] == cluster_id:
                return cluster
        return {"cluster_id": cluster_id, "nodes": [], "size": 0}

    def _build_graph_analysis(self) -> None:
        """Precompute clustering and centrality analytics at startup."""
        if self._undirected_graph.number_of_nodes() == 0:
            self._graph_analysis = {
                "overview": {
                    "total_clusters": 0,
                    "modularity": 0.0,
                    "largest_cluster_size": 0,
                    "top_degree_nodes": [],
                    "top_pagerank_nodes": [],
                    "top_betweenness_nodes": [],
                },
                "clusters": [],
            }
            return

        communities = nx.community.louvain_communities(self._undirected_graph, seed=42)
        modularity = nx.community.modularity(self._undirected_graph, communities) if communities else 0.0

        degree_scores = nx.degree_centrality(self._undirected_graph)
        pagerank_scores = nx.pagerank(self.G, alpha=0.85)
        betweenness_scores = nx.betweenness_centrality(
            self._undirected_graph,
            k=min(250, self._undirected_graph.number_of_nodes()),
            normalized=True,
            seed=42,
        )

        clusters: list[dict[str, Any]] = []
        for cluster_id, community in enumerate(sorted(communities, key=len, reverse=True), start=1):
            node_ids = sorted(community)
            type_breakdown = Counter(self.G.nodes[node_id].get("node_type", "UNKNOWN") for node_id in node_ids)
            sample_nodes = [self._build_ranked_node(node_id, None) for node_id in node_ids[:8]]
            top_central_nodes = sorted(
                node_ids,
                key=lambda node_id: degree_scores.get(node_id, 0.0),
                reverse=True,
            )[:5]
            clusters.append({
                "cluster_id": cluster_id,
                "size": len(node_ids),
                "node_ids": node_ids,
                "node_type_breakdown": dict(type_breakdown),
                "sample_nodes": sample_nodes,
                "top_central_nodes": [
                    self._build_ranked_node(node_id, degree_scores.get(node_id, 0.0))
                    for node_id in top_central_nodes
                ],
            })

        self._graph_analysis = {
            "overview": {
                "total_clusters": len(clusters),
                "modularity": round(float(modularity), 4),
                "largest_cluster_size": clusters[0]["size"] if clusters else 0,
                "top_degree_nodes": self._top_ranked_nodes(degree_scores, top_n=8),
                "top_pagerank_nodes": self._top_ranked_nodes(pagerank_scores, top_n=8),
                "top_betweenness_nodes": self._top_ranked_nodes(betweenness_scores, top_n=8),
            },
            "clusters": clusters,
        }

    def _top_ranked_nodes(self, scores: dict[str, float], top_n: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return [self._build_ranked_node(node_id, score) for node_id, score in ranked]

    def _build_ranked_node(self, node_id: str, score: float | None) -> dict[str, Any]:
        attrs = dict(self.G.nodes[node_id])
        payload = {
            "id": node_id,
            "node_type": attrs.get("node_type", "UNKNOWN"),
            "label": self._node_label(node_id, attrs),
        }
        if score is not None:
            payload["score"] = round(float(score), 6)
        return payload

    def _node_label(self, node_id: str, attrs: dict[str, Any]) -> str:
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

    def _build_semantic_index(self) -> None:
        """Generate a normalized embedding for each node at startup."""
        node_ids: list[str] = []
        corpus: list[str] = []

        for node_id, attrs in self.G.nodes(data=True):
            node_ids.append(node_id)
            corpus.append(self._node_to_semantic_text(node_id, attrs))

        if not corpus:
            return

        self._semantic_node_ids = node_ids
        self._semantic_embeddings = self._embedding_model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _build_lexical_index(self) -> None:
        """Build a lightweight token index for robust lexical matching."""
        for node_id, attrs in self.G.nodes(data=True):
            node_type = attrs.get("node_type", node_id.split(":", 1)[0])
            text_parts = [node_id, node_type]
            for value in attrs.values():
                if isinstance(value, (str, int, float)):
                    text_parts.append(str(value))
            tokens = re.findall(r"[A-Za-z0-9]+", " ".join(text_parts).lower())
            if not tokens:
                continue
            self._lexical_tokens.update(tokens)
            self._lexical_tokens_by_type[node_type].update(tokens)

    def _node_to_semantic_text(self, node_id: str, attrs: dict[str, Any]) -> str:
        """Build a readable text representation of a node for embedding."""
        node_type = attrs.get("node_type", node_id.split(":", 1)[0])
        label = (
            attrs.get("name")
            or attrs.get("description")
            or attrs.get("plantName")
            or attrs.get("material")
            or attrs.get("customer_id")
            or attrs.get("product_id")
            or attrs.get("salesOrder_id")
            or attrs.get("billingDocument_id")
            or attrs.get("deliveryDocument_id")
            or node_id.split(":", 1)[1]
        )

        parts = [
            f"{node_type}: {label}",
            f"node id: {node_id}",
            f"o2c concepts: {self._node_type_semantic_context(node_type)}",
        ]
        for key, value in attrs.items():
            if key == "node_type":
                continue
            rendered = self._render_semantic_value(value)
            if rendered:
                parts.append(f"{key}: {rendered}")

        neighbour_context = self._render_neighbourhood_context(node_id)
        if neighbour_context:
            parts.append(f"graph context: {neighbour_context}")

        return ", ".join(parts)

    def _render_semantic_value(self, value: Any) -> str:
        """Flatten nested node properties into text for embedding."""
        if value is None:
            return ""
        if isinstance(value, dict):
            parts = []
            for key, nested in value.items():
                rendered = self._render_semantic_value(nested)
                if rendered:
                    parts.append(f"{key} {rendered}")
            return ", ".join(parts)
        if isinstance(value, list):
            parts = [self._render_semantic_value(item) for item in value]
            return ", ".join(part for part in parts if part)
        return str(value)

    def _node_type_semantic_context(self, node_type: str) -> str:
        """Add O2C-specific aliases to improve semantic recall."""
        aliases = {
            "Customer": "customer sold-to party account receivable buyer",
            "Product": "product material sku item sold",
            "Plant": "plant warehouse shipping location fulfillment site",
            "SalesOrder": "sales order customer order demand order-to-cash",
            "SalesOrderItem": "sales order line item material requested quantity",
            "DeliveryDocument": "delivery shipment goods movement fulfillment dispatch",
            "DeliveryItem": "delivery line shipped quantity warehouse movement",
            "BillingDocument": "billing invoice receivable customer invoice document",
            "BillingItem": "invoice line billed product billed quantity",
            "JournalEntry": "journal entry accounting document accounts receivable clearing document posting",
            "Payment": "payment cash receipt settlement clearing incoming payment",
        }
        return aliases.get(node_type, node_type.lower())

    def _render_neighbourhood_context(self, node_id: str, max_edges: int = 6) -> str:
        """Summarize a small amount of graph context for semantic indexing."""
        snippets: list[str] = []

        for source, _, edge_attrs in self.G.in_edges(node_id, data=True):
            source_type = self.G.nodes[source].get("node_type", source.split(":", 1)[0])
            edge_type = edge_attrs.get("edge_type", "connected_from")
            snippets.append(f"from {source_type} via {edge_type}")
            if len(snippets) >= max_edges:
                return ", ".join(snippets)

        for _, target, edge_attrs in self.G.out_edges(node_id, data=True):
            target_type = self.G.nodes[target].get("node_type", target.split(":", 1)[0])
            edge_type = edge_attrs.get("edge_type", "connected_to")
            snippets.append(f"to {target_type} via {edge_type}")
            if len(snippets) >= max_edges:
                break

        return ", ".join(snippets)
