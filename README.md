# DodgeAI O2C — LLM-Powered Graph Query System

**Live Demo:** [<FRONTEND_URL>  ](https://dodge-ai-o2c.vercel.app/)
**Repo:** https://github.com/harshbatra01/dodgeAI-o2c  

---

## Overview

This project converts a fragmented SAP Order-to-Cash dataset into a unified graph and exposes it through an LLM-powered conversational interface.

The system supports:
- Natural language queries → structured graph operations  
- Data-grounded answers (no hallucination)  
- Graph visualization + highlighting  
- Advanced graph analysis  

The implementation is **output-first** — the final user experience drives architecture and execution.

---

## What This System Delivers

### ✅ Core Requirements (Fully Implemented)

#### Graph Construction
- Nodes and edges modeled from SAP O2C dataset  
- Proper entity and relationship types  
- Stored as JSON and loaded into NetworkX  

#### Graph Visualization
- Full graph rendered with `react-force-graph`  
- Node metadata inspection  
- Clear UI separation  

#### Conversational Query Interface
- Natural language → structured query plan  
- Data-grounded responses  
- Streaming responses via SSE  

#### Guardrails
- Off-topic prompts rejected  
- Dataset-related queries always allowed  

---

### 🚀 Bonus Features (Fully Implemented)

#### Conversation Memory
- Resolves “this”, “that”, “those” across turns  
- Works across chat and graph  

#### Highlighting + Graph Focus
- Chat returns `referenced_nodes`  
- Graph highlights relevant entities  
- Multi-node auto-fit view  

#### Semantic Search
- Sentence-transformers (`all-MiniLM-L6-v2`)  
- `/graph/semantic-search` endpoint  
- Frontend highlighting  

#### Graph Clustering / Analysis
- Cluster detection  
- Top hubs + modularity  
- Interactive UI  

#### LLM-First Query Handling
- Deterministic routing (simple queries)  
- LLM fallback (complex queries)  
- Handles messy natural language  

---

## Example Queries

### Aggregation
- "Which products are associated with the highest number of billing documents?"
- "Tell me the top 5 customers"
- "Which customers contribute most to billing"

### Flow / Tracing
- "Trace the full flow of billing document 90504248"
- "How does a billing document move through the system?"

### Conditional / Missing Links
- "Show invoices without payments"
- "Find incomplete flows"
- "Delivered but not billed"

### Natural Language
- "umm what does this customer usually buy the most"
- "what does this customer mostly interact with"
- "what has jordan been billed the most for"

---

## Architecture (High-Level)

### Backend (FastAPI + NetworkX)
- `graph.json` → in-memory graph  
- Query plans executed via structured engine  
- LLM generates plans, engine executes safely  
- Limits + timeouts for reliability  

### Frontend (React + Vite)
- Force graph visualization  
- Chat with streaming SSE  
- Search + clustering panels  
- Clean responsive layout  

---

## Tech Stack

### Backend
- FastAPI  
- NetworkX  
- Sentence-Transformers (MiniLM)  
- NumPy  
- Uvicorn  

### Frontend
- React + Vite  
- react-force-graph  
- TypeScript  

---

## Deployment

### Render (Backend)
**Start Command:**
```bash
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 10000
