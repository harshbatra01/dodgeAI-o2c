"""
graph.py — Endpoints for navigating and querying the NetworkX graph
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Request

from graph_store import GraphStore

router = APIRouter(prefix="/graph", tags=["Graph Explorer"])

# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    nodes_by_type: dict[str, int]
    edges_by_type: dict[str, int]

class NodeModel(BaseModel):
    id: str
    node_type: str = "UNKNOWN"
    incomplete: Optional[bool] = None
    # Catch-all dict for the rest of properties
    model_config = {
        "extra": "allow"
    }

class EdgeModel(BaseModel):
    source: str
    target: str
    edge_type: str
    model_config = {
        "extra": "allow"
    }

class GraphResponse(BaseModel):
    nodes: list[NodeModel]
    edges: list[EdgeModel]

class SearchResponse(BaseModel):
    results: list[NodeModel]
    total_hits: int

class RankedNodeModel(BaseModel):
    id: str
    node_type: str
    label: str
    score: float | None = None

class ClusterSummaryModel(BaseModel):
    cluster_id: int
    size: int
    node_type_breakdown: dict[str, int]
    sample_nodes: list[RankedNodeModel]
    top_central_nodes: list[RankedNodeModel]
    node_ids: list[str] = Field(default_factory=list)

class ClusterListResponse(BaseModel):
    clusters: list[ClusterSummaryModel]
    total_clusters: int

class GraphAnalysisOverview(BaseModel):
    total_clusters: int
    modularity: float
    largest_cluster_size: int
    top_degree_nodes: list[RankedNodeModel]
    top_pagerank_nodes: list[RankedNodeModel]
    top_betweenness_nodes: list[RankedNodeModel]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _get_store(request: Request) -> GraphStore:
    """Helper to extract the GraphStore from app state."""
    return request.app.state.store


@router.get("/stats", response_model=GraphStats)
def get_graph_stats(request: Request) -> Any:
    """Get node/edge counts and grouping metadata for the graph."""
    store = _get_store(request)
    return store.get_stats()


@router.get("/node/{node_id:path}", response_model=GraphResponse)
def get_node(request: Request, node_id: str) -> Any:
    """Returns a specific node plus its direct (1-hop) neighbors and connections."""
    store = _get_store(request)
    if not store.has_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        
    result = store.get_node_with_neighbours(node_id)
    return result


@router.get("/subgraph", response_model=GraphResponse)
def get_subgraph(
    request: Request,
    center: str = Query(..., description="The ID of the central node"),
    depth: int = Query(2, description="Hop distance (radius) for ego graph", ge=1, le=5)
) -> Any:
    """Get the ego graph encompassing a specified radius around a central node."""
    store = _get_store(request)
    if not store.has_node(center):
        raise HTTPException(status_code=404, detail=f"Center node '{center}' not found")

    result = store.get_ego_graph(center, radius=depth)
    return result


@router.get("/search", response_model=SearchResponse)
def search_graph(
    request: Request,
    q: str = Query(..., min_length=1, description="Text fragment to search across nodes")
) -> Any:
    """Find a list of nodes whose attributes match the search substring."""
    store = _get_store(request)
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    results = store.search_nodes(q, limit=20)
    return {
        "results": results,
        "total_hits": len(results)
    }


@router.get("/semantic-search", response_model=SearchResponse)
def semantic_search_graph(
    request: Request,
    q: str = Query(..., min_length=1, description="Natural-language semantic search query")
) -> Any:
    """Find the top semantically similar nodes using sentence embeddings."""
    store = _get_store(request)
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    results = store.semantic_search_nodes(q, limit=10)
    return {
        "results": results,
        "total_hits": len(results)
    }


@router.get("/analysis/overview", response_model=GraphAnalysisOverview)
def get_graph_analysis_overview(request: Request) -> Any:
    """Return graph clustering and centrality overview metrics."""
    store = _get_store(request)
    return store.get_graph_analysis_overview()


@router.get("/analysis/clusters", response_model=ClusterListResponse)
def get_graph_clusters(
    request: Request,
    limit: int = Query(10, ge=1, le=25, description="Number of cluster summaries to return")
) -> Any:
    """Return summaries for the largest graph communities."""
    store = _get_store(request)
    return store.get_graph_clusters(limit=limit)


@router.get("/analysis/clusters/{cluster_id}", response_model=ClusterSummaryModel)
def get_graph_cluster_detail(request: Request, cluster_id: int) -> Any:
    """Return full detail for a specific cluster."""
    store = _get_store(request)
    result = store.get_cluster_detail(cluster_id)
    if result.get("size", 0) == 0 and not result.get("node_ids"):
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return result


@router.get("/all", response_model=GraphResponse)
def get_all_graph(request: Request) -> Any:
    """Return the entire dataset of nodes and edges."""
    store = _get_store(request)
    nodes = []
    for nid, attrs in store.G.nodes(data=True):
        nodes.append({"id": nid, **attrs})
        
    edges = []
    for u, v, attrs in store.G.edges(data=True):
        edges.append({"source": u, "target": v, **attrs})
        
    return {
        "nodes": nodes,
        "edges": edges
    }
