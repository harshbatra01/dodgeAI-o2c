export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  nodes_by_type: Record<string, number>;
  edges_by_type: Record<string, number>;
}

export interface NodeModel {
  id: string;
  node_type: string;
  incomplete?: boolean | null;
  [key: string]: any; // Allows extra fields
}

export interface EdgeModel {
  source: string;
  target: string;
  edge_type: string;
  [key: string]: any;
}

export interface GraphResponse {
  nodes: NodeModel[];
  edges: EdgeModel[];
}

export interface SearchResponse {
  results: NodeModel[];
  total_hits: number;
}

export interface RankedNode {
  id: string;
  node_type: string;
  label: string;
  score?: number | null;
}

export interface ClusterSummary {
  cluster_id: number;
  size: number;
  node_type_breakdown: Record<string, number>;
  sample_nodes: RankedNode[];
  top_central_nodes: RankedNode[];
  node_ids: string[];
}

export interface ClusterListResponse {
  clusters: ClusterSummary[];
  total_clusters: number;
}

export interface GraphAnalysisOverview {
  total_clusters: number;
  modularity: number;
  largest_cluster_size: number;
  top_degree_nodes: RankedNode[];
  top_pagerank_nodes: RankedNode[];
  top_betweenness_nodes: RankedNode[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface MemoryTurn {
  referenced_nodes: string[];
  user_message?: string;
  assistant_message?: string;
}

export interface ConversationMemory {
  turns: MemoryTurn[];
}

export interface ChatRequest {
  messages: ChatMessage[];
  memory?: ConversationMemory;
}

export interface ChatResponse {
  reply: string;
  query_plan?: Record<string, any>;
  data?: Record<string, any>;
}

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) || 'http://localhost:8000';

export const apiClient = {
  getStats: async (): Promise<GraphStats> => {
    const res = await fetch(`${BASE_URL}/graph/stats`);
    if (!res.ok) throw new Error('Failed to fetch stats');
    return res.json();
  },

  getAll: async (): Promise<GraphResponse> => {
    const res = await fetch(`${BASE_URL}/graph/all`);
    if (!res.ok) throw new Error('Failed to fetch entire graph');
    return res.json();
  },

  getNode: async (nodeId: string): Promise<GraphResponse> => {
    const res = await fetch(`${BASE_URL}/graph/node/${encodeURIComponent(nodeId)}`);
    if (!res.ok) throw new Error('Failed to fetch node');
    return res.json();
  },

  getSubgraph: async (center: string, depth: number = 2): Promise<GraphResponse> => {
    const res = await fetch(`${BASE_URL}/graph/subgraph?center=${encodeURIComponent(center)}&depth=${depth}`);
    if (!res.ok) throw new Error('Failed to fetch subgraph');
    return res.json();
  },

  searchGraph: async (query: string): Promise<SearchResponse> => {
    const res = await fetch(`${BASE_URL}/graph/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error('Failed to search graph');
    return res.json();
  },

  semanticSearchGraph: async (query: string): Promise<SearchResponse> => {
    const res = await fetch(`${BASE_URL}/graph/semantic-search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error('Failed to run semantic search');
    return res.json();
  },

  getGraphAnalysisOverview: async (): Promise<GraphAnalysisOverview> => {
    const res = await fetch(`${BASE_URL}/graph/analysis/overview`);
    if (!res.ok) throw new Error('Failed to fetch graph analysis overview');
    return res.json();
  },

  getGraphClusters: async (limit: number = 10): Promise<ClusterListResponse> => {
    const res = await fetch(`${BASE_URL}/graph/analysis/clusters?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch graph clusters');
    return res.json();
  },

  getGraphClusterDetail: async (clusterId: number): Promise<ClusterSummary> => {
    const res = await fetch(`${BASE_URL}/graph/analysis/clusters/${clusterId}`);
    if (!res.ok) throw new Error('Failed to fetch graph cluster detail');
    return res.json();
  },

  postChat: async (request: ChatRequest): Promise<ChatResponse> => {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) throw new Error('Chat failed');
    return res.json();
  },

  streamChat: async function* (request: ChatRequest): AsyncGenerator<any, void, unknown> {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!res.ok) throw new Error('Chat streaming failed');
    if (!res.body) throw new Error('ReadableStream not supported');

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep the last incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr.trim() === '[DONE]') {
              return;
            }
            try {
              const parsed = JSON.parse(dataStr);
              yield parsed;
            } catch (e) {
              // Ignore partial JSON chunks
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },
};
