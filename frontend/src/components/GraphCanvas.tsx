import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { X, Maximize, Network, Sparkles } from 'lucide-react';
import { ClusterSummary, GraphAnalysisOverview, NodeModel, RankedNode, apiClient, GraphResponse } from '../api/client';

const NODE_COLORS: Record<string, string> = {
  SalesOrder: '#3b82f6', // blue
  DeliveryDocument: '#10b981', // emerald
  BillingDocument: '#f59e0b', // amber
  JournalEntry: '#8b5cf6', // violet
  Customer: '#f43f5e', // rose
  Product: '#06b6d4', // cyan
  Plant: '#64748b', // slate
  Payment: '#22c55e', // green
  SalesOrderItem: '#60a5fa', // lighter blue
  DeliveryItem: '#34d399', // lighter emerald
  BillingItem: '#fbbf24', // lighter amber
  DEFAULT: '#a3a3a3' // neutral gray
};

interface GraphCanvasProps {
  onNodeClick?: (node: NodeModel) => void;
  highlightedNodes?: string[];
  chatHighlightMode?: 'focus' | 'expand';
  searchHighlightedNodes?: string[];
  showLabels: boolean;
  focusRequest?: { id: string; token: number } | null;
  chatFocusRequest?: { ids: string[]; token: number } | null;
}

export function GraphCanvas({
  onNodeClick,
  highlightedNodes,
  chatHighlightMode = 'focus',
  searchHighlightedNodes = [],
  showLabels,
  focusRequest,
  chatFocusRequest,
}: GraphCanvasProps) {
  const fgRef = useRef<any>();
  const canvasRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<NodeModel | null>(null);
  const [nodeDetails, setNodeDetails] = useState<GraphResponse | null>(null);
  const [graphAnalysis, setGraphAnalysis] = useState<GraphAnalysisOverview | null>(null);
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<ClusterSummary | null>(null);
  const [analysisHighlightedNodes, setAnalysisHighlightedNodes] = useState<string[]>([]);
  const [highlightedClusterId, setHighlightedClusterId] = useState<number | null>(null);
  const [isAnalysisLoading, setIsAnalysisLoading] = useState(true);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [panelHeights, setPanelHeights] = useState({ details: 42, analysis: 38, legend: 20 });

  type ResizeHandle = 'details-analysis' | 'analysis-legend';
  const [dragState, setDragState] = useState<{
    handle: ResizeHandle;
    startY: number;
    start: { details: number; analysis: number; legend: number };
    sidebarHeight: number;
    hasDetails: boolean;
  } | null>(null);

  useEffect(() => {
    const initialize = async () => {
      try {
        const fullGraph = await apiClient.getAll();
        setGraphData({
          nodes: fullGraph.nodes.map(n => ({ ...n })),
          links: fullGraph.edges.map(e => ({ ...e }))
        });
        const [overview, clusterResponse] = await Promise.all([
          apiClient.getGraphAnalysisOverview(),
          apiClient.getGraphClusters(8),
        ]);
        setGraphAnalysis(overview);
        setClusters(clusterResponse.clusters);
        if (clusterResponse.clusters.length > 0) {
          setSelectedClusterId(clusterResponse.clusters[0].cluster_id);
          setSelectedCluster(clusterResponse.clusters[0]);
        }
      } catch (e) {
        console.error("Failed to init graph", e);
        setAnalysisError('Graph analysis unavailable');
      } finally {
        setIsAnalysisLoading(false);
      }
    };
    initialize();
  }, []);

  useEffect(() => {
    const observeTarget = canvasRef.current;
    if (!observeTarget) return;
    
    // Set initial size
    setDimensions({ width: observeTarget.clientWidth, height: observeTarget.clientHeight });
    
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries[0] && entries[0].contentRect) {
        setDimensions({ 
          width: entries[0].contentRect.width, 
          height: entries[0].contentRect.height 
        });
      }
    });
    
    resizeObserver.observe(observeTarget);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const handleMove = (event: MouseEvent) => {
      if (!dragState) return;
      const minPanelPx = 120;
      const minDetailsPx = 170;
      const total = Math.max(1, dragState.sidebarHeight);
      const deltaPx = event.clientY - dragState.startY;
      const px = {
        details: (dragState.start.details / 100) * total,
        analysis: (dragState.start.analysis / 100) * total,
        legend: (dragState.start.legend / 100) * total,
      };

      if (dragState.handle === 'details-analysis' && dragState.hasDetails) {
        let nextDetails = px.details + deltaPx;
        let nextAnalysis = px.analysis - deltaPx;
        const maxDetails = total - px.legend - minPanelPx;
        nextDetails = Math.min(Math.max(nextDetails, minDetailsPx), maxDetails);
        nextAnalysis = total - px.legend - nextDetails;
        setPanelHeights({
          details: (nextDetails / total) * 100,
          analysis: (nextAnalysis / total) * 100,
          legend: (px.legend / total) * 100,
        });
      }

      if (dragState.handle === 'analysis-legend') {
        let nextAnalysis = px.analysis + deltaPx;
        let nextLegend = px.legend - deltaPx;
        const minTop = dragState.hasDetails ? minPanelPx : minPanelPx;
        const maxAnalysis = total - (dragState.hasDetails ? px.details : 0) - minPanelPx;
        nextAnalysis = Math.min(Math.max(nextAnalysis, minTop), maxAnalysis);
        nextLegend = total - (dragState.hasDetails ? px.details : 0) - nextAnalysis;
        setPanelHeights({
          details: dragState.hasDetails ? (px.details / total) * 100 : 0,
          analysis: (nextAnalysis / total) * 100,
          legend: (nextLegend / total) * 100,
        });
      }
    };

    const handleUp = () => setDragState(null);

    if (dragState) {
      window.addEventListener('mousemove', handleMove);
      window.addEventListener('mouseup', handleUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [dragState]);

  useEffect(() => {
    if (selectedNode) {
      setPanelHeights(prev => (prev.details < 30 ? { details: 42, analysis: 38, legend: 20 } : prev));
    } else {
      setPanelHeights({ details: 0, analysis: 78, legend: 22 });
    }
  }, [selectedNode]);

  const handleNodeClick = useCallback(async (node: any) => {
    setSelectedNode(node);
    if (onNodeClick) onNodeClick(node);
    
    try {
      const details = await apiClient.getNode(node.id);
      setNodeDetails(details);
    } catch (e) {
      console.error(e);
    }

    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 1000);
      fgRef.current.zoom(4, 1000);
    }
  }, [onNodeClick]);

  const focusNodeById = useCallback((nodeId: string) => {
    const targetNode = graphData.nodes.find(node => node.id === nodeId);
    if (targetNode) {
      void handleNodeClick(targetNode);
    }
  }, [graphData.nodes, handleNodeClick]);

  const focusNodesByIds = useCallback((nodeIds: string[]) => {
    const uniqueIds = Array.from(new Set(nodeIds.filter(Boolean)));
    if (uniqueIds.length === 0) return;
    const uniqueIdSet = new Set(uniqueIds);

    const targetNodes = graphData.nodes.filter((node) => uniqueIdSet.has(node.id));
    if (targetNodes.length === 0) return;

    if (targetNodes.length === 1) {
      const targetNode = targetNodes[0];
      setSelectedNode(targetNode);
      if (onNodeClick) onNodeClick(targetNode);
      void apiClient
        .getNode(targetNode.id)
        .then((details) => setNodeDetails(details))
        .catch((error) => console.error(error));

      if (fgRef.current) {
        fgRef.current.centerAt(targetNode.x, targetNode.y, 850);
        fgRef.current.zoom(3.2, 850);
      }
      return;
    }

    setSelectedNode(targetNodes[0]);
    if (onNodeClick) onNodeClick(targetNodes[0]);
    void apiClient
      .getNode(targetNodes[0].id)
      .then((details) => setNodeDetails(details))
      .catch((error) => console.error(error));

    if (fgRef.current) {
      fgRef.current.zoomToFit(900, 70, (node: any) => uniqueIdSet.has(node.id));
    }
  }, [graphData.nodes, onNodeClick]);

  useEffect(() => {
    if (!focusRequest) return;
    focusNodeById(focusRequest.id);
  }, [focusNodeById, focusRequest]);

  useEffect(() => {
    if (!chatFocusRequest) return;
    focusNodesByIds(chatFocusRequest.ids);
  }, [chatFocusRequest, focusNodesByIds]);

  const chatHighlightedNodeSet = useMemo(() => {
    const ids = new Set((highlightedNodes ?? []).filter(Boolean));
    if (ids.size === 0) return ids;
    if (chatHighlightMode === 'focus') return ids;

    for (const link of graphData.links) {
      const sourceId = typeof link.source === 'string' ? link.source : link.source?.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target?.id;
      if (!sourceId || !targetId) continue;
      if (ids.has(sourceId) || ids.has(targetId)) {
        ids.add(sourceId);
        ids.add(targetId);
      }
    }
    return ids;
  }, [chatHighlightMode, graphData.links, highlightedNodes]);

  const handleClusterChange = useCallback(async (clusterId: number) => {
    setSelectedClusterId(clusterId);
    try {
      const cluster = await apiClient.getGraphClusterDetail(clusterId);
      setSelectedCluster(cluster);
      if (highlightedClusterId !== null) {
        setAnalysisHighlightedNodes(cluster.node_ids);
        setHighlightedClusterId(cluster.cluster_id);
      }
    } catch (error) {
      console.error(error);
      setAnalysisError('Failed to load cluster detail');
    }
  }, [highlightedClusterId]);

  const handleHighlightCluster = useCallback(() => {
    if (!selectedCluster) return;
    const isActive = highlightedClusterId === selectedCluster.cluster_id;
    if (isActive) {
      setAnalysisHighlightedNodes([]);
      setHighlightedClusterId(null);
      return;
    }
    setAnalysisHighlightedNodes(selectedCluster.node_ids);
    setHighlightedClusterId(selectedCluster.cluster_id);
  }, [highlightedClusterId, selectedCluster]);

  const renderRankedNodeList = useCallback((title: string, nodes: RankedNode[]) => {
    if (nodes.length === 0) return null;
    return (
      <div className="analysis-section">
        <div className="analysis-section-title">{title}</div>
        <div className="analysis-node-list">
          {nodes.slice(0, 5).map((node) => (
            <button
              key={node.id}
              type="button"
              className="analysis-node-button"
              onClick={() => focusNodeById(node.id)}
            >
              <span className="analysis-node-label">{node.label}</span>
              <span className="analysis-node-meta">
                {node.node_type}
                {node.score !== undefined && node.score !== null ? ` · ${node.score.toFixed(4)}` : ''}
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  }, [focusNodeById]);

  const handleFitView = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 50);
    }
  };

  const closeInspector = () => {
    setSelectedNode(null);
    setNodeDetails(null);
  };

  const renderNodeDetails = () => {
    if (!selectedNode) return null;
    
    const excludeKeys = new Set(['id', 'node_type', 'x', 'y', 'vx', 'vy', 'fx', 'fy', 'index', 'color', 'incomplete']);
    const fields = Object.entries(selectedNode).filter(([k]) => !excludeKeys.has(k) && !k.startsWith('_'));
    const renderFields = fields.slice(0, 8);
    const hiddenCount = fields.length - 8;

    return (
      <div className="inspector-card">
        <div className="inspector-header">
          <div className="inspector-title">
            <span 
              className="inspector-color-dot" 
              style={{ backgroundColor: NODE_COLORS[selectedNode.node_type] || NODE_COLORS.DEFAULT }}
            />
            <span title={selectedNode.id} className="inspector-type">
              {selectedNode.node_type}
            </span>
          </div>
          <button onClick={closeInspector} className="inspector-close">
            <X size={16} />
          </button>
        </div>
        
        <div className="inspector-body">
          <div className="inspector-id">{selectedNode.id}</div>
          
          <div className="inspector-fields">
            {renderFields.map(([key, value]) => (
              <div key={key} className="inspector-field">
                <div className="field-key">{key}</div>
                <div className="field-value">
                  {value === null || value === undefined ? <span className="field-null">null</span> : String(value)}
                </div>
              </div>
            ))}
            
            {hiddenCount > 0 && (
              <div className="hidden-fields-msg">
                {hiddenCount} additional fields hidden for readability
              </div>
            )}
          </div>
        </div>

        <div className="inspector-footer">
          <span>Connections: {nodeDetails ? nodeDetails.edges.length : '...'}</span>
        </div>
      </div>
    );
  };

  const isSelectedClusterHighlighted =
    selectedCluster !== null && highlightedClusterId === selectedCluster.cluster_id;

  return (
    <div className="graph-container">
      <div className="graph-main" ref={canvasRef}>
        {/* Tools */}
        <div className="graph-tools">
          <button onClick={handleFitView} className="tool-btn">
            <Maximize size={16} /> Fit View
          </button>
        </div>

        <ForceGraph2D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeLabel={(node: any) => node.id}
          nodeColor={(node: any) => NODE_COLORS[node.node_type] || NODE_COLORS.DEFAULT}
          nodeRelSize={6}
          linkColor={() => 'rgba(255,255,255,0.1)'}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          onNodeClick={handleNodeClick}
          backgroundColor="#0a0a0a"
          nodeCanvasObject={(node: any, ctx, globalScale) => {
            const isChatHighlighted = chatHighlightedNodeSet.has(node.id);
            const isSemanticHighlighted = searchHighlightedNodes.includes(node.id);
            const isAnalysisHighlighted = analysisHighlightedNodes.includes(node.id);

            if (isSemanticHighlighted || isChatHighlighted || isAnalysisHighlighted) {
              ctx.beginPath();
              ctx.arc(node.x, node.y, 10, 0, 2 * Math.PI, false);
              ctx.fillStyle = isSemanticHighlighted
                ? 'rgba(250, 204, 21, 0.9)'
                : isAnalysisHighlighted
                  ? 'rgba(34, 211, 238, 0.85)'
                  : 'rgba(255, 255, 255, 0.8)';
              ctx.shadowColor = isSemanticHighlighted
                ? '#facc15'
                : isAnalysisHighlighted
                  ? '#22d3ee'
                  : 'white';
              ctx.shadowBlur = 10;
              ctx.fill();
              ctx.shadowBlur = 0;
            }

            const isIncomplete = node.incomplete === true;
            const color = NODE_COLORS[node.node_type] || NODE_COLORS.DEFAULT;

            ctx.beginPath();
            ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI, false);

            if (isIncomplete) {
              ctx.fillStyle = color;
              ctx.globalAlpha = 0.4;
              ctx.fill();
              ctx.globalAlpha = 1.0;

              ctx.beginPath();
              ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI, false);
              ctx.lineWidth = 1.5;
              ctx.strokeStyle = color;
              ctx.setLineDash([3, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
            } else {
              ctx.fillStyle = color;
              ctx.fill();
            }

            if (showLabels) {
              const label = node.id.split(':')[1] || node.id;
              const fontSize = 12/globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              const textWidth = ctx.measureText(label).width;
              const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

              ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
              ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, bckgDimensions[0], bckgDimensions[1]);

              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = NODE_COLORS[node.node_type] || NODE_COLORS.DEFAULT;
              ctx.fillText(label, node.x, node.y);
              node.__bckgDimensions = bckgDimensions;
            }
          }}
        />
      </div>

      <aside className={`graph-sidebar ${selectedNode ? 'has-details' : 'no-details'}`} ref={sidebarRef}>
        {selectedNode && (
          <>
            <section
              className="sidebar-section sidebar-section-details"
              style={{ flexBasis: `${panelHeights.details}%` }}
            >
              {renderNodeDetails()}
            </section>
            <button
              type="button"
              className="sidebar-resize-handle"
              onMouseDown={(event) => {
                event.preventDefault();
                const sidebar = sidebarRef.current;
                if (!sidebar) return;
                setDragState({
                  handle: 'details-analysis',
                  startY: event.clientY,
                  start: panelHeights,
                  sidebarHeight: sidebar.clientHeight,
                  hasDetails: true,
                });
              }}
              aria-label="Resize Node Details and Graph Analysis panels"
            >
              <span />
            </button>
          </>
        )}

        <section
          className="sidebar-section sidebar-section-analysis"
          style={{ flexBasis: `${selectedNode ? panelHeights.analysis : panelHeights.analysis + panelHeights.details}%` }}
        >
        <div className="graph-analysis-card">
        <div className="graph-analysis-header">
          <div className="graph-analysis-title">
            <Network size={15} />
            <span>Graph Analysis</span>
          </div>
          {graphAnalysis && <span className="graph-analysis-badge">{graphAnalysis.total_clusters} clusters</span>}
        </div>

        {isAnalysisLoading && <div className="graph-analysis-empty">Analyzing graph structure...</div>}
        {!isAnalysisLoading && analysisError && <div className="graph-analysis-empty">{analysisError}</div>}

        {!isAnalysisLoading && !analysisError && graphAnalysis && (
          <>
            <div className="analysis-overview-grid">
              <div className="analysis-overview-tile">
                <div className="analysis-overview-label">Modularity</div>
                <div className="analysis-overview-value">{graphAnalysis.modularity.toFixed(4)}</div>
              </div>
              <div className="analysis-overview-tile">
                <div className="analysis-overview-label">Largest Cluster</div>
                <div className="analysis-overview-value">{graphAnalysis.largest_cluster_size}</div>
              </div>
            </div>

            <div className="analysis-section">
              <div className="analysis-section-title">Communities</div>
              <div className="analysis-cluster-controls">
                <select
                  className="analysis-cluster-select"
                  value={selectedClusterId ?? ''}
                  onChange={(event) => handleClusterChange(Number(event.target.value))}
                >
                  {clusters.map((cluster) => (
                    <option key={cluster.cluster_id} value={cluster.cluster_id}>
                      Cluster {cluster.cluster_id} · {cluster.size} nodes
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className={`analysis-cluster-highlight ${isSelectedClusterHighlighted ? 'active' : ''}`}
                  onClick={handleHighlightCluster}
                >
                  <Sparkles size={14} /> {isSelectedClusterHighlighted ? 'Unhighlight' : 'Highlight'}
                </button>
              </div>
              {selectedCluster && (
                <div className="analysis-cluster-summary">
                  <div className="analysis-cluster-breakdown">
                    {Object.entries(selectedCluster.node_type_breakdown).slice(0, 4).map(([nodeType, count]) => (
                      <span key={nodeType} className="analysis-chip">{nodeType}: {count}</span>
                    ))}
                  </div>
                  <div className="analysis-node-list">
                    {selectedCluster.top_central_nodes.slice(0, 4).map((node) => (
                      <button
                        key={node.id}
                        type="button"
                        className="analysis-node-button"
                        onClick={() => focusNodeById(node.id)}
                      >
                        <span className="analysis-node-label">{node.label}</span>
                        <span className="analysis-node-meta">{node.node_type}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {renderRankedNodeList('Top Hubs', graphAnalysis.top_degree_nodes)}
            {renderRankedNodeList('Top PageRank', graphAnalysis.top_pagerank_nodes)}
            {renderRankedNodeList('Top Bridges', graphAnalysis.top_betweenness_nodes)}
          </>
        )}
      </div>
        </section>

        <button
          type="button"
          className="sidebar-resize-handle"
          onMouseDown={(event) => {
            event.preventDefault();
            const sidebar = sidebarRef.current;
            if (!sidebar) return;
            setDragState({
              handle: 'analysis-legend',
              startY: event.clientY,
              start: panelHeights,
              sidebarHeight: sidebar.clientHeight,
              hasDetails: Boolean(selectedNode),
            });
          }}
          aria-label="Resize Graph Analysis and Node Types panels"
        >
          <span />
        </button>

        <section className="sidebar-section sidebar-section-legend" style={{ flexBasis: `${panelHeights.legend}%` }}>
        <div className="graph-legend-card">
          <div className="legend-title">Node Types</div>
          <div className="legend-items">
            {Object.entries(NODE_COLORS).filter(([k]) => k !== 'DEFAULT' && !k.includes('Item') && k !== 'Payment').map(([type, color]) => (
              <div key={type} className="legend-item">
                <span className="legend-color" style={{ backgroundColor: color }}></span>
                <span className="legend-label" title={type}>{type}</span>
              </div>
            ))}
          </div>
        </div>
        </section>
      </aside>
    </div>
  );
}
