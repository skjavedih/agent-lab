# 🤖 Agent Lab — How It Works

> **A deep-dive into the full execution flow:** from a user's text message → LLM reasoning → tool selection → Qdrant vector DB retrieval / Tavily web search / pandas analytics → final response.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component Breakdown](#3-component-breakdown)
4. [End-to-End Message Flow](#4-end-to-end-message-flow)
5. [Deep Dive: RAG Pipeline (Qdrant + nomic-embed-text)](#5-deep-dive-rag-pipeline-qdrant--nomic-embed-text)
6. [Deep Dive: What Happens in Docker?](#6-deep-dive-what-happens-in-docker)
7. [Deep Dive: Tavily Web Search](#7-deep-dive-tavily-web-search)
8. [Deep Dive: Sandboxed Pandas Analytics](#8-deep-dive-sandboxed-pandas-analytics)
9. [Key Functions & Call Graph](#9-key-functions--call-graph)
10. [Data Flow: CSV → Qdrant Vector DB](#10-data-flow-csv--qdrant-vector-db)
11. [Environment & Configuration](#11-environment--configuration)
12. [Technology Stack Summary](#12-technology-stack-summary)

---

## 1. System Overview

Agent Lab is a **locally-hosted AI assistant** specialized in IPL cricket data (2008–2017). It uses the **ReAct (Reason + Act)** agentic loop, where the LLM iterates through:

```
THINK → ACT (call tool) → OBSERVE (get result) → repeat → ANSWER
```

All inference runs **100% locally** via [Ollama](https://ollama.com) — no cloud LLM API needed. The only external service is an optional [Tavily](https://tavily.com) API for live web search.

### Key Design Principles

| Principle | Implementation |
|---|---|
| Local-first LLM | Ollama (`qwen3:8b`) runs on your machine |
| Semantic search | Qdrant vector DB (in Docker) stores narrative match summaries |
| Structured analytics | Pandas DataFrames queried via LLM-generated code |
| Live web search | Tavily API (optional, API key required) |
| Mode-aware output | `text` mode → rich markdown; `voice` mode → short plain speech |
| Front-end agnostic | `run_turn()` service function works for CLI, voice, or HTTP API |

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                               │
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐ │
│   │   main.py (CLI)  │    │  app/voice.py    │    │  Future: FastAPI/    │ │
│   │  text or voice   │    │  (STT → TTS)     │    │  WebSocket server    │ │
│   └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────────┘ │
│            │                       │                        │               │
│            └───────────────────────┴────────────────────────┘              │
│                                    │ user_text, thread_id, mode            │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER  (app/service.py)                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  run_turn(user_text, thread_id, mode)  →  Iterator[AgentEvent]      │   │
│  │                                                                     │   │
│  │  • Manages in-memory thread conversation history                    │   │
│  │  • Caches compiled agent graphs per mode                            │   │
│  │  • Streams events: thought_action | observation | answer            │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AGENT LAYER  (app/agent.py)                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  build_agent(mode)  →  CompiledStateGraph                           │   │
│  │                                                                     │   │
│  │  LangGraph ReAct Loop:                                              │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │   │
│  │  │  LLM Node    │───▶│  Tool Node   │───▶│  LLM Node (again)    │  │   │
│  │  │  (Reason)    │◀───│  (Execute)   │    │  (Observe & Answer)  │  │   │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘  │   │
│  │                                                                     │   │
│  │  ChatOllama (qwen3:8b) ←→ Ollama HTTP API (localhost:11434)         │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOLS LAYER  (app/tools.py)                          │
│                                                                             │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│   │ calculator  │  │get_current   │  │  web_search  │  │  search_docs   │ │
│   │             │  │_time         │  │  (Tavily)    │  │  (Qdrant RAG)  │ │
│   └─────────────┘  └──────────────┘  └──────┬───────┘  └───────┬────────┘ │
│                                             │                   │           │
│                                    ┌────────┴──────┐   ┌───────┴─────────┐ │
│                                    │  Tavily API   │   │  query_ipl_stats│ │
│                                    │  (internet)   │   │  (analytics.py) │ │
│                                    └───────────────┘   └───────┬─────────┘ │
└────────────────────────────────────────────────────────────────┼────────────┘
                                                                  │
                    ┌─────────────────────────────────────────────┤
                    ▼                                             ▼
┌──────────────────────────────────┐     ┌────────────────────────────────────┐
│     RAG LAYER  (app/rag.py)      │     │   ANALYTICS (app/analytics.py)     │
│                                  │     │                                    │
│  search(query, k=4)              │     │  query_ipl_stats(question)         │
│  1. embed query with             │     │  1. LLM generates pandas snippet   │
│     nomic-embed-text             │     │  2. Sandboxed exec() against       │
│     → vector[768 floats]         │     │     matches.csv + deliveries.csv   │
│  2. Cosine similarity search     │     │  3. Returns scalar/DataFrame result│
│     in Qdrant collection         │     │                                    │
│  3. Return top-k docs +          │     │  DataFrames loaded ONCE at import  │
│     similarity scores            │     │  (636 matches, 150k rows)          │
└──────────────┬───────────────────┘     └────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 DOCKER: Qdrant Vector Database                                │
│                 Container: qdrant  |  Image: qdrant/qdrant:latest            │
│                                                                              │
│   Port 6333 (HTTP API) ←── QdrantClient  ←── langchain-qdrant               │
│   Port 6334 (gRPC)                                                           │
│                                                                              │
│   Collection: ipl_matches                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Point ID (UUID5)  │  Vector [768 floats]  │  Payload (metadata)   │   │
│   │────────────────────┼───────────────────────┼───────────────────────│   │
│   │  match_1_chunk_0   │  [0.12, -0.33, ...]   │  match_id, season,    │   │
│   │  match_2_chunk_0   │  [0.44, 0.11, ...]    │  teams, venue, date   │   │
│   │  ...636 matches... │  ...                  │  ...                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Persisted Volume: ./qdrant_data → /qdrant/storage (inside container)       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### `main.py` — CLI Entry Point
- Parses `--mode` (`text`/`voice`) and `--thread` arguments
- Verifies Ollama is reachable via HTTP `/api/tags`
- Loops on `input()`, delegates each turn to `run_turn()`
- Prints streamed `AgentEvent` objects to console

### `app/service.py` — Service Orchestrator
- **`run_turn()`** — the single public function; returns a lazy `Iterator[AgentEvent]`
- Maintains `_THREAD_STORES` — an in-memory dict of `thread_id → List[BaseMessage]`
- Caches compiled agent graphs in `_COMPILED_AGENTS` (so graph is only built once per mode)
- Translates raw LangGraph stream chunks into typed `AgentEvent` objects:
  - `AIMessage` with `tool_calls` → `thought_action` event
  - `ToolMessage` → `observation` event
  - `AIMessage` without `tool_calls` → `answer` event

### `app/agent.py` — LLM + ReAct Graph Builder
- **`get_ollama_llm()`** — creates `ChatOllama` pointing at `http://localhost:11434`
- **`build_agent()`** — compiles LangGraph `CompiledStateGraph` via `langchain.agents.create_agent`
- Selects system prompt based on mode:
  - `TEXT_SYSTEM_PROMPT` — detailed routing rules for IPL data, markdown output
  - `VOICE_SYSTEM_PROMPT` — TTS-friendly, no markdown, 2-3 sentences max
- **`strip_thinking()`** — strips `<think>...</think>` reasoning blocks from Qwen3's output

### `app/tools.py` — Tool Definitions (5 tools)

| Tool | Trigger Condition | Implementation |
|---|---|---|
| `calculator` | Pure arithmetic | Safe AST evaluator, whitelist-only operators |
| `get_current_time` | Time/date questions | `datetime.now().astimezone()` |
| `web_search` | Live/external facts | Tavily API client, top-3 results |
| `search_docs` | Match narrative questions | `app.rag.search()` → Qdrant semantic search |
| `query_ipl_stats` | Aggregations/counts | `app.analytics.query_ipl_stats()` → LLM-generated pandas |

### `app/rag.py` — Vector DB Pipeline
- **`NomicEmbeddingsWrapper`** — adds `search_document:` / `search_query:` prefixes required by `nomic-embed-text`
- **`build_match_documents()`** — transforms raw CSV rows into rich narrative paragraphs
- **`ingest()`** — upserts all documents into Qdrant collection
- **`search()`** — embeds query and performs cosine similarity search

### `app/analytics.py` — Sandboxed Pandas Engine
- Loads `matches.csv` and `deliveries.csv` into module-level DataFrames (parsed once at import)
- **`query_ipl_stats()`** — sends question to LLM with a template prompt, gets back Python code, executes in a locked-down namespace

### `scripts/ingest.py` — One-time Data Loader
- CLI wrapper around `app.rag.ingest()`
- Run once to populate Qdrant: `python -m scripts.ingest [--reset]`

### `docker-compose.yml` — Infrastructure
- Runs Qdrant as a Docker container
- Binds host `./qdrant_data` → `/qdrant/storage` inside container for persistence

---

## 4. End-to-End Message Flow

Here is the step-by-step execution trace for the example query:

> **"Tell me about the 2016 final match"**

```
Step 1: User types message
──────────────────────────
main.py → input("You: ") = "Tell me about the 2016 final match"

Step 2: run_turn() called
──────────────────────────
app/service.py:run_turn(
    user_text = "Tell me about the 2016 final match",
    thread_id = "default",
    mode = "text"
)

Step 3: Retrieve (or build) agent graph
────────────────────────────────────────
service.py:get_agent("text")
  └── If not cached: agent.py:build_agent(mode="text")
        ├── get_ollama_llm()  →  ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")
        └── create_agent(model=llm, tools=[...5 tools...], system_prompt=TEXT_SYSTEM_PROMPT)
            └── Returns CompiledStateGraph (LangGraph ReAct graph)

Step 4: Append user message to thread history
──────────────────────────────────────────────
_THREAD_STORES["default"].append(HumanMessage("Tell me about the 2016 final match"))

Step 5: Stream the LangGraph graph
─────────────────────────────────────
agent.stream({"messages": thread_history}, config={"configurable": {"thread_id": "default"}})

  ┌─────────────────────────────────────────────────────────────┐
  │  LangGraph Node: "agent" (LLM call)                         │
  │                                                             │
  │  Sends to Ollama API:                                       │
  │  POST http://localhost:11434/api/chat                       │
  │  Body: {                                                    │
  │    "model": "qwen3:8b",                                     │
  │    "messages": [                                            │
  │      {"role": "system", "content": TEXT_SYSTEM_PROMPT},    │
  │      {"role": "user", "content": "Tell me about the 2016   │
  │                                    final match"}            │
  │    ],                                                       │
  │    "tools": [search_docs schema, query_ipl_stats schema...] │
  │  }                                                          │
  │                                                             │
  │  Ollama (qwen3:8b) reasons → decides to call search_docs   │
  │  Returns AIMessage with tool_calls: [{                      │
  │    "name": "search_docs",                                   │
  │    "args": {"query": "2016 IPL final match"}                │
  │  }]                                                         │
  └────────────────────────────┬────────────────────────────────┘
                               │  YIELD → AgentEvent(type="thought_action")
                               │  main.py prints: [THOUGHT/ACTION] Call tool: search_docs(...)
                               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  LangGraph Node: "tools" (tool execution)                   │
  │                                                             │
  │  tools.py:search_docs(query="2016 IPL final match")         │
  │    └── rag.py:search(query="2016 IPL final match", k=4)     │
  │          ├── NomicEmbeddingsWrapper.embed_query()           │
  │          │     └── Ollama: nomic-embed-text model           │
  │          │           POST http://localhost:11434/api/embed  │
  │          │           Input: "search_query: 2016 IPL final"  │
  │          │           Output: vector[768 floats]             │
  │          │                                                  │
  │          └── QdrantVectorStore.similarity_search_with_score │
  │                GET http://localhost:6333/collections/       │
  │                    ipl_matches/points/search                │
  │                {                                            │
  │                  "vector": [0.12, -0.33, 0.45, ...],        │
  │                  "limit": 4,                                │
  │                  "with_payload": true                       │
  │                }                                            │
  │                → Returns top-4 documents by cosine score   │
  │                                                             │
  │  Formats result as XML retrieved_document chunks            │
  │  Returns ToolMessage with the match narrative text          │
  └────────────────────────────┬────────────────────────────────┘
                               │  YIELD → AgentEvent(type="observation")
                               │  main.py prints: [OBSERVATION] Tool 'search_docs' returned: ...
                               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  LangGraph Node: "agent" (LLM call again — final answer)    │
  │                                                             │
  │  Sends to Ollama with full conversation including           │
  │  tool result observation → LLM synthesizes final answer     │
  │                                                             │
  │  Returns AIMessage (no tool_calls) → final answer text      │
  └────────────────────────────┬────────────────────────────────┘
                               │  YIELD → AgentEvent(type="answer")
                               │  main.py prints: [ANSWER] ...

Step 6: Print elapsed time
───────────────────────────
main.py prints: [⏱️ Response time: 24.21s]
```

---

## 5. Deep Dive: RAG Pipeline (Qdrant + nomic-embed-text)

### Phase A: Ingestion (one-time setup)

```
python -m scripts.ingest
        │
        ▼
scripts/ingest.py:main()
        │
        ▼
app/rag.py:ingest(reset=False)
        │
        ├── 1. Connect to Qdrant at http://localhost:6333
        │         (running inside Docker container)
        │
        ├── 2. Create collection "ipl_matches" if not exists
        │         Vector size: 768  (nomic-embed-text output dim)
        │         Distance metric: Cosine similarity
        │
        ├── 3. build_match_documents()
        │         ├── pd.read_csv("data/matches.csv")    → 636 rows
        │         ├── pd.read_csv("data/deliveries.csv") → 150,460 rows
        │         │
        │         ├── Aggregate per match:
        │         │     • deliveries.groupby(match_id, inning, batting_team)
        │         │       → innings totals ("Innings 1 (RCB) scored 227 runs")
        │         │     • deliveries.groupby(match_id, batsman)
        │         │       → top scorer per match
        │         │     • deliveries[dismissal_kind in BOWLER_WICKET_KINDS]
        │         │       → best bowler per match (most wickets, lowest economy)
        │         │
        │         ├── Build narrative paragraph per match:
        │         │     "Match 580 of the 2016 IPL season was played on 2016-04-12
        │         │      at M Chinnaswamy Stadium, Bangalore between RCB and SRH.
        │         │      SRH won the toss and elected to field first.
        │         │      Innings 1 (RCB) scored 227 runs. Innings 2 (SRH) scored 182 runs.
        │         │      RCB won by 45 runs. Player of the Match was AB de Villiers.
        │         │      Top scorer was AB de Villiers with 82 runs.
        │         │      Best bowler was Mustafizur Rahman taking 2 wickets (economy: 6.50)."
        │         │
        │         └── TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        │               Splits long summaries into token-safe chunks
        │               Each chunk keeps metadata: match_id, teams, season, venue, date
        │
        └── 4. Embed & Upload in batches of 50
                  For each chunk:
                    NomicEmbeddingsWrapper.embed_documents(["search_document: <text>"])
                      → POST http://localhost:11434/api/embed (nomic-embed-text)
                      → Returns vector[768 floats]
                  
                  IDs are deterministic UUID5:
                    uuid.uuid5(NAMESPACE_DNS, "match_580_chunk_0")
                    (prevents duplicates on re-runs)
                  
                  vector_store.add_documents(batch_docs, ids=batch_ids)
                    → POST http://localhost:6333/collections/ipl_matches/points
```

### Phase B: Query-time Retrieval

```
search_docs(query="2016 final match maximum score")
        │
        ▼
app/rag.py:search(query, k=4)
        │
        ├── 1. NomicEmbeddingsWrapper.embed_query(query)
        │         Prepends "search_query: " prefix
        │         → POST http://localhost:11434/api/embed
        │         → vector[768 floats]
        │
        ├── 2. QdrantVectorStore.similarity_search_with_score(query, k=4)
        │         Qdrant performs HNSW approximate nearest neighbour search
        │         Returns top-4 vectors by cosine similarity
        │         Each result: (Document, float_score)
        │
        └── 3. Format output: List[{text, metadata, score}]
```

### Why `search_document:` and `search_query:` prefixes?

`nomic-embed-text` is an **asymmetric embedding model** trained with task prefixes. Without the correct prefix, the embedding space shifts and retrieval quality drops significantly:
- `search_document:` → used when indexing (push docs into Qdrant)
- `search_query:` → used when searching (embed user question)

---

## 6. Deep Dive: What Happens in Docker?

Docker's role in this project is **minimal but essential** — it runs only the **Qdrant vector database**. Everything else (Python app, Ollama LLM) runs natively on your host machine.

### `docker-compose.yml` Explained

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest      # Official Qdrant image from Docker Hub
    container_name: qdrant           # Named "qdrant" for easy reference
    ports:
      - "6333:6333"                  # HTTP REST API (host:container)
      - "6334:6334"                  # gRPC API (used by some clients)
    volumes:
      - ./qdrant_data:/qdrant/storage   # Persistent storage mount
    restart: unless-stopped          # Auto-restart unless manually stopped
```

### What Does Docker Actually Store?

```
your-machine/
└── agent-lab/
    └── qdrant_data/              ← HOST directory (you see this in your file system)
        └── collection/
            └── ipl_matches/
                ├── segments/     ← HNSW graph index files
                ├── payload_storage/  ← Metadata (match_id, teams, etc.)
                └── wal/          ← Write-ahead log for durability

Inside Docker Container:
    /qdrant/storage/              ← CONTAINER path (mounted from above)
        └── collection/
            └── ipl_matches/
                └── ...same files...
```

> **Key insight:** The data lives on your **host machine** in `./qdrant_data`. Docker is just the process running Qdrant's server. If you delete the container, your indexed data is safe. If you delete `qdrant_data/`, you lose all indexed vectors and must re-run ingestion.

### Network Communication

```
Python app (host)                    Docker container
─────────────────                    ────────────────
QdrantClient("http://localhost:6333")
         │
         │  HTTP REST requests
         │  via Docker port mapping
         ▼
    localhost:6333  ──────────────▶  container:6333
                                     Qdrant HTTP server
                                     (serves vector search API)
```

### Docker Lifecycle

```bash
# Start Qdrant (must be running before ingestion or queries)
docker compose up -d

# Check it's healthy
curl http://localhost:6333/healthz

# View collection info
curl http://localhost:6333/collections/ipl_matches

# Stop (data persists in qdrant_data/)
docker compose down

# Full reset (wipe data too)
docker compose down && rm -rf qdrant_data/
```

---

## 7. Deep Dive: Tavily Web Search

When the LLM detects the question is about **real-world, current, or external** information, it routes to `web_search`:

```
tools.py:web_search(query)
        │
        ├── 1. Read TAVILY_API_KEY from .env
        │         If not set → returns error string (agent sees this and tries another approach)
        │
        ├── 2. TavilyClient(api_key).search(query, max_results=3, search_depth="basic")
        │         Makes HTTPS request to Tavily's search API
        │         Tavily crawls the web and returns curated, snippet-rich results
        │
        └── 3. Format: "[1] Title\nURL: ...\nSnippet: ..."
                  Returns top-3 results as single formatted string
                  LLM reads this as an observation and synthesizes the answer
```

**Routing examples:**

| User Question | Tool Selected | Reason |
|---|---|---|
| "What is today's date?" | `get_current_time` | Not external info |
| "Latest cricket news" | `web_search` via Tavily | Needs live internet |
| "Who won the 2016 IPL final?" | `search_docs` (Qdrant) | Historical IPL match data |
| "Most runs in IPL 2016?" | `query_ipl_stats` | Aggregate calculation |

---

## 8. Deep Dive: Sandboxed Pandas Analytics

For **numerical, aggregation, or ranking** IPL questions:

```
tools.py:query_ipl_stats(question)
        │
        ▼
app/analytics.py:query_ipl_stats(question)
        │
        ├── 1. Module-level DataFrames (loaded once at import)
        │         matches: 636 rows × 18 columns (from matches.csv)
        │         deliveries: 150,460 rows × 21 columns (from deliveries.csv)
        │
        ├── 2. LLM Code Generation
        │         Sends PANDAS_TRANSLATION_PROMPT to qwen3:8b via Ollama
        │         Prompt contains: DataFrame schemas, 5 worked examples, user question
        │         LLM responds with Python code like:
        │           result = int((matches['winner'] == 'Mumbai Indians').sum())
        │
        ├── 3. Token sanitization (security layer)
        │         Checks for forbidden tokens:
        │           __import__, import , open(, exec(, eval(, compile(, os., sys., subprocess
        │         If any found → returns Security Error (code never executes)
        │
        ├── 4. Sandboxed exec()
        │         Executes code in restricted namespace:
        │         local_namespace = {
        │             "__builtins__": {len, int, float, str, ...safe primitives only},
        │             "pd": pandas,
        │             "matches": matches_df,
        │             "deliveries": deliveries_df
        │         }
        │         No access to: os, sys, file system, network, subprocess
        │
        └── 5. Return result
                  If result is DataFrame/Series → .to_string()
                  Else → str(result)
                  Returned as ToolMessage observation to the LLM
```

---

## 9. Key Functions & Call Graph

```
main.py
└── main()
    ├── check_ollama_service(base_url, model_name)
    └── run_turn(user_text, thread_id, mode)           ← app/service.py

app/service.py
└── run_turn(user_text, thread_id, mode)
    ├── get_agent(mode)                                ← cached agent graph
    │   └── build_agent(mode)                         ← app/agent.py
    │       ├── get_ollama_llm(model, base_url, ...)
    │       └── create_agent(model, tools, system_prompt)
    └── agent.stream({"messages": history}, config)
        yields: AIMessage (tool_call)  → AgentEvent("thought_action")
                ToolMessage            → AgentEvent("observation")
                AIMessage (final)      → AgentEvent("answer")

app/tools.py
├── calculator(expression)
│   └── _safe_eval_node(ast.parse(expression))
├── get_current_time()
│   └── datetime.now().astimezone().strftime(...)
├── web_search(query)
│   └── TavilyClient.search(query, max_results=3)
├── search_docs(query)
│   └── app/rag.py:search(query, k=4)
│       ├── NomicEmbeddingsWrapper.embed_query(query)
│       │   └── OllamaEmbeddings.embed_query("search_query: " + query)
│       └── QdrantVectorStore.similarity_search_with_score(query, k=4)
└── query_ipl_stats(question)
    └── app/analytics.py:query_ipl_stats(question)
        ├── get_ollama_llm().invoke([HumanMessage(prompt)])  → code string
        └── exec(code, local_namespace)  → result

app/rag.py
├── ingest(reset)
│   ├── build_match_documents()
│   │   ├── pd.read_csv(matches.csv, deliveries.csv)
│   │   ├── groupby aggregations (innings, top scorer, best bowler)
│   │   └── TokenTextSplitter.split_text(paragraph)
│   └── vector_store.add_documents(docs, ids=uuid5_ids)
└── search(query, k)
    ├── get_embeddings() → NomicEmbeddingsWrapper
    ├── get_qdrant_client() → QdrantClient("http://localhost:6333")
    └── vector_store.similarity_search_with_score(query, k)

scripts/ingest.py
└── main() → app/rag.ingest(reset)
```

---

## 10. Data Flow: CSV → Qdrant Vector DB

```
data/matches.csv (636 rows)           data/deliveries.csv (150,460 rows)
        │                                         │
        └──────────────┬──────────────────────────┘
                       │ pd.read_csv()
                       ▼
             app/rag.py: build_match_documents()
                       │
                       ├── groupby(match_id, inning, batting_team) → innings totals
                       ├── groupby(match_id, batsman) → top scorer
                       ├── groupby(match_id, bowler) + dismissal filter → best bowler
                       │
                       ▼
             636 narrative paragraphs (one per match)
                       │
                       ▼
             TokenTextSplitter(chunk_size=500, chunk_overlap=50)
             (chunks very long summaries, keeps most as single chunks)
                       │
                       ▼
             ~636+ Document chunks with metadata:
             {
               page_content: "Match 580 of the 2016...",
               metadata: {
                 match_id: 580, teams: "RCB vs SRH",
                 season: 2016, venue: "...", date: "2016-04-12",
                 chunk_index: 0
               }
             }
                       │
                       ▼ NomicEmbeddingsWrapper.embed_documents()
                         prefix added: "search_document: <text>"
                         POST http://localhost:11434/api/embed (nomic-embed-text)
                       │
                       ▼
             vector[768 floats] per chunk
                       │
                       ▼ QdrantVectorStore.add_documents() in batches of 50
                         ID: uuid5("match_580_chunk_0") — deterministic, idempotent
                       │
                       ▼
             ┌──────────────────────────────────────────┐
             │  Qdrant Collection: "ipl_matches"        │
             │  (running in Docker on port 6333)        │
             │  persisted at: ./qdrant_data/            │
             └──────────────────────────────────────────┘
```

---

## 11. Environment & Configuration

All settings are loaded from `.env` via `python-dotenv`. The `.env.example` shows all keys:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | LLM model for reasoning & analytics code generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP API endpoint |
| `ENABLE_THINKING` | `false` | Enable/disable Qwen3's `<think>` reasoning tokens |
| `TAVILY_API_KEY` | `your_key_here` | API key for Tavily live web search |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB endpoint (Docker container) |
| `QDRANT_COLLECTION` | `ipl_matches` | Qdrant collection name |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for RAG |
| `CHUNK_SIZE` | `500` | Max tokens per document chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |

---

## 12. Technology Stack Summary

| Layer | Technology | Version | Role |
|---|---|---|---|
| **LLM** | Ollama + qwen3:8b | — | Local inference, tool calling, code generation |
| **Embeddings** | Ollama + nomic-embed-text | — | Text → 768-dim vectors |
| **Agent Framework** | LangChain / LangGraph | 1.4.2 / 1.2.12 | ReAct graph, tool binding, message state |
| **Vector DB** | Qdrant | latest | Stores & searches embedded match documents |
| **Container** | Docker + docker-compose | — | Runs Qdrant with persistent volume |
| **Data Analysis** | Pandas | 3.0.6 | Aggregations on matches.csv & deliveries.csv |
| **Web Search** | Tavily Python SDK | 0.8.4 | Real-time internet search |
| **Python** | Python 3.12+ | 3.12 | Runtime |
| **Config** | python-dotenv | 1.2.3 | `.env` file management |

---

## Quick-Start Commands

```bash
# 1. Start Qdrant vector DB (Docker must be running)
docker compose up -d

# 2. Pull Ollama models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 3. Ingest IPL data into Qdrant (one-time, takes ~5–10 minutes)
python -m scripts.ingest

# 4. Run the agent
python main.py

# --- Utility commands ---

# Reset & re-ingest Qdrant from scratch
python -m scripts.ingest --reset

# Inspect raw vector retrieval (debug RAG without LLM)
python -m scripts.inspect_retrieval "Who won the 2016 IPL final?"

# Run in voice mode (short, no markdown)
python main.py --mode voice

# Use a named conversation thread
python main.py --thread session-42
```

---

*Generated for Agent Lab — ReAct IPL Cricket Assistant*
