import { useState, useRef, useEffect } from 'react';
import { Send, Settings, User, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient, ChatMessage, MemoryTurn } from '../api/client';

export interface ChatPanelProps {
  onNodesReferenced?: (nodes: string[]) => void;
  onClose?: () => void;
}

export function ChatPanel({ onNodesReferenced, onClose }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [memoryTurns, setMemoryTurns] = useState<MemoryTurn[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Setup the initial empty assistant message block
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
      
      const stream = apiClient.streamChat({ messages: newMessages, memory: { turns: memoryTurns } });
      let currentReferencedNodes: string[] = [];
      let finalAssistantContent = '';
      for await (const chunk of stream) {
        setIsLoading(false); // as soon as we got first chunk, stop the typing indicator dot dot dot
        
        if (chunk.token) {
          finalAssistantContent += chunk.token;
          setMessages(prev => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            copy[lastIdx] = { 
              ...copy[lastIdx], 
              content: copy[lastIdx].content + chunk.token 
            };
            return copy;
          });
        } else if (chunk.error) {
          finalAssistantContent += `\n\n**Error:** ${chunk.error}`;
          setMessages(prev => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            copy[lastIdx] = { 
              ...copy[lastIdx], 
              content: copy[lastIdx].content + `\n\n**Error:** ${chunk.error}` 
            };
            return copy;
          });
        } else if (chunk.referenced_nodes) {
          currentReferencedNodes = chunk.referenced_nodes;
        }
      }

      if (onNodesReferenced) {
        onNodesReferenced(currentReferencedNodes);
      }

      setMemoryTurns(prev => [
        ...prev,
        {
          referenced_nodes: currentReferencedNodes,
          user_message: userMessage,
          assistant_message: finalAssistantContent,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection error. Please ensure the backend is running.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div>
          <h2>Chat with Graph</h2>
          <p className="subheading">Order to Cash</p>
        </div>
        {onClose && (
          <button type="button" className="chat-close-btn" onClick={onClose} aria-label="Close chat panel">
            <X size={16} />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <Settings size={48} className="empty-icon" />
            <p>Agent is ready. Ask questions about the SAP O2C dataset, trace document flows, or aggregate data.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message-row ${msg.role}`}>
            <div className={`avatar ${msg.role}`}>
              {msg.role === 'user' ? <User size={16} /> : <Settings size={16} />}
            </div>
            <div className={`bubble ${msg.role}`}>
              {msg.role === 'assistant' ? (
                <div className="markdown-content">
                  {msg.content ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    isLoading && idx === messages.length - 1 && (
                      <div className="typing-indicator">
                        <span></span><span></span><span></span>
                      </div>
                    )
                  )}
                </div>
              ) : (
                msg.content.split('\n').map((line, i) => (
                  <span key={i}>
                    {line}
                    {i < msg.content.split('\n').length - 1 && <br />}
                  </span>
                ))
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="chat-input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <textarea 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Ask anything about the data..."
            disabled={isLoading}
            className="chat-textarea"
            rows={1}
          />
          <button 
            type="submit" 
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            <Send size={18} />
          </button>
        </form>
        
        {/* Status Line */}
        <div className="status-line">
          <span className={`status-dot ${isLoading ? 'processing' : 'ready'}`}></span>
          {isLoading ? 'Graph Agent is processing...' : 'Graph Agent is awaiting instructions'}
        </div>
      </div>
    </div>
  );
}
