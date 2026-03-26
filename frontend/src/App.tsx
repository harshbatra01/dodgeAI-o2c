import { useEffect, useMemo, useRef, useState } from 'react';
import { MessageSquare, Search, Tag } from 'lucide-react';
import { NodeModel, apiClient } from './api/client';
import { GraphCanvas } from './components/GraphCanvas';
import { ChatPanel } from './components/ChatPanel';

function App() {
  const [highlightedNodes, setHighlightedNodes] = useState<string[]>([]);
  const [showLabels, setShowLabels] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem('o2c-chat-open');
      return stored === null ? true : stored === '1';
    } catch {
      return true;
    }
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<NodeModel[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [focusRequest, setFocusRequest] = useState<{ id: string; token: number } | null>(null);
  const [chatFocusRequest, setChatFocusRequest] = useState<{ ids: string[]; token: number } | null>(null);
  const [searchHighlightedNodes, setSearchHighlightedNodes] = useState<string[]>([]);
  const searchShellRef = useRef<HTMLDivElement>(null);
  const highlightTimeoutRef = useRef<number | null>(null);
  const focusTokenRef = useRef(0);
  const chatFocusTokenRef = useRef(0);

  const trimmedQuery = useMemo(() => searchQuery.trim(), [searchQuery]);

  useEffect(() => {
    try {
      window.sessionStorage.setItem('o2c-chat-open', isChatOpen ? '1' : '0');
    } catch {
      // noop
    }
  }, [isChatOpen]);

  useEffect(() => {
    const query = trimmedQuery;
    if (!query) {
      setSearchResults([]);
      setSearchError(null);
      setIsSearchLoading(false);
      setShowSearchDropdown(false);
      return;
    }

    setSearchError(null);
    setIsSearchLoading(true);

    const timeoutId = window.setTimeout(async () => {
      try {
        const response = await apiClient.semanticSearchGraph(query);
        setSearchResults(response.results);
        setShowSearchDropdown(true);
      } catch (error) {
        console.error(error);
        setSearchResults([]);
        setSearchError('Search failed');
        setShowSearchDropdown(true);
      } finally {
        setIsSearchLoading(false);
      }
    }, 220);

    return () => window.clearTimeout(timeoutId);
  }, [trimmedQuery]);

  useEffect(() => {
    const handleOutsideClick = (event: MouseEvent) => {
      if (!searchShellRef.current) return;
      if (!searchShellRef.current.contains(event.target as Node)) {
        setShowSearchDropdown(false);
      }
    };
    window.addEventListener('mousedown', handleOutsideClick);
    return () => window.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  useEffect(() => {
    if (searchHighlightedNodes.length === 0) return;
    if (highlightTimeoutRef.current) window.clearTimeout(highlightTimeoutRef.current);
    highlightTimeoutRef.current = window.setTimeout(() => {
      setSearchHighlightedNodes([]);
      highlightTimeoutRef.current = null;
    }, 5000);
    return () => {
      if (highlightTimeoutRef.current) {
        window.clearTimeout(highlightTimeoutRef.current);
      }
    };
  }, [searchHighlightedNodes]);

  useEffect(() => {
    return () => {
      if (highlightTimeoutRef.current) window.clearTimeout(highlightTimeoutRef.current);
    };
  }, []);

  const getNodeDisplayLabel = (node: NodeModel) =>
    node.name ||
    node.description ||
    node.plantName ||
    node.material ||
    node.customer_id ||
    node.product_id ||
    node.salesOrder_id ||
    node.billingDocument_id ||
    node.deliveryDocument_id ||
    node.accountingDocument_id ||
    node.id.split(':').slice(1).join(':');

  const handleNodesReferenced = (nodes: string[]) => {
    const deduped = Array.from(new Set((nodes || []).filter(Boolean)));
    if (deduped.length === 0) {
      setHighlightedNodes([]);
      return;
    }
    setHighlightedNodes(deduped);
    chatFocusTokenRef.current += 1;
    setChatFocusRequest({ ids: deduped, token: chatFocusTokenRef.current });
  };

  const focusNodeFromSearch = (nodeId: string) => {
    focusTokenRef.current += 1;
    setFocusRequest({ id: nodeId, token: focusTokenRef.current });
    setSearchHighlightedNodes([nodeId]);
    setShowSearchDropdown(false);
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <div className="header-left">
          <span className="breadcrumb">Mapping › <strong>Order to Cash</strong></span>
        </div>
        <div className="header-right">
          <div className="header-search-shell" ref={searchShellRef}>
            <div className="header-search-input-wrap">
              <Search size={15} className="header-search-icon" />
              <input
                className="header-search-input"
                type="text"
                value={searchQuery}
                onChange={(event) => {
                  setSearchQuery(event.target.value);
                  setShowSearchDropdown(Boolean(event.target.value.trim()));
                  setSearchError(null);
                }}
                onFocus={() => {
                  if (trimmedQuery) setShowSearchDropdown(true);
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Escape') {
                    setShowSearchDropdown(false);
                  }
                }}
                placeholder="Search entities"
              />
            </div>
            {showSearchDropdown && (
              <div className="header-search-dropdown">
                {searchError && <div className="header-search-empty">{searchError}</div>}
                {!searchError && isSearchLoading && <div className="header-search-empty">Searching...</div>}
                {!searchError && !isSearchLoading && searchResults.length === 0 && (
                  <div className="header-search-empty">No entities found for "{trimmedQuery}"</div>
                )}
                {!searchError && !isSearchLoading && searchResults.map((result) => (
                  <button
                    key={result.id}
                    type="button"
                    className="header-search-result"
                    onClick={() => focusNodeFromSearch(result.id)}
                  >
                    <span className="header-search-result-type">{result.node_type}</span>
                    <span className="header-search-result-label">{getNodeDisplayLabel(result)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            className={`header-btn ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels((prev) => !prev)}
          >
            <Tag size={14} />
            <span>Show Labels</span>
          </button>

          <button
            type="button"
            className={`header-btn chat-toggle-btn ${isChatOpen ? 'active' : ''}`}
            onClick={() => setIsChatOpen((prev) => !prev)}
          >
            <MessageSquare size={14} />
            <span>Chat</span>
          </button>
        </div>
      </div>

      <div className={`main-content ${isChatOpen ? 'chat-open' : 'chat-closed'}`}>
        <div className="graph-panel">
          <GraphCanvas
            highlightedNodes={highlightedNodes}
            chatHighlightMode="focus"
            searchHighlightedNodes={searchHighlightedNodes}
            showLabels={showLabels}
            focusRequest={focusRequest}
            chatFocusRequest={chatFocusRequest}
          />
        </div>

        <div className={`chat-area ${isChatOpen ? 'open' : 'closed'}`}>
          <ChatPanel onNodesReferenced={handleNodesReferenced} onClose={() => setIsChatOpen(false)} />
        </div>
      </div>
    </div>
  );
}

export default App;
