# Agent Lab — Architecture Diagrams

Session-ready architecture for Agent Lab: overview plus five focused areas (Python → chatbot → Ollama, RAG, chunking/Qdrant, Tavily, voice).

---

## Connecting your chatbot (start here)

Use this section when you run the **browser UI** and want to see how a message travels through the stack.

### 1. What must be running

```mermaid
flowchart LR
    subgraph You["You"]
        BROWSER["Browser\nlocalhost:8000"]
    end

    subgraph Python["Python process"]
        UV["uvicorn app.server:app"]
        SVC["app/service.py\nrun_turn()"]
        UV --> SVC
    end

    subgraph Required["Required for chat"]
        OLLAMA["Ollama :11434\nqwen3:8b"]
    end

    subgraph Optional["Optional by question type"]
        QDRANT["Qdrant :6333\nDocker compose"]
        TAVILY["Tavily API\n.env key"]
        JEV["Jev / TypeSafe\nrouting"]
    end

    BROWSER -->|"GET /"| UV
    BROWSER -->|"POST /chat SSE"| UV
    SVC --> OLLAMA
    SVC -.->|ipl narrative| QDRANT
    SVC -.->|live web| TAVILY
    SVC -.->|classify| JEV
```

| Step | Command / check |
|------|------------------|
| Ollama | Desktop app or `ollama serve`; models `qwen3:8b` (+ `nomic-embed-text` for RAG) |
| Qdrant (RAG) | `docker compose up -d` then `python -m scripts.ingest` once |
| Web chatbot | `uvicorn app.server:app --reload` → open **http://127.0.0.1:8000/** |
| Terminal only | `python main.py` (skips FastAPI; calls `run_turn()` directly) |

**Important:** `main.py` and the Web UI are two **front doors** into the same `run_turn()` function. Only the Web UI needs Uvicorn.

### 2. Web UI — one message, full path

```mermaid
sequenceDiagram
    autonumber
    participant You as You (browser)
    participant HTML as static/index.html
    participant API as app/server.py
    participant SVC as service.run_turn
    participant Mem as _THREAD_STORES[thread_id]
    participant R as router.route
    participant LLM as Ollama qwen3:8b
    participant Tool as tools (RAG / stats / web)
    participant Agent as LangGraph ReAct

    You->>HTML: Open http://127.0.0.1:8000/
    API-->>HTML: GET / serves index.html

    You->>HTML: Type message, Send
    HTML->>API: POST /chat JSON { text, thread_id, mode }
    Note over HTML: thread_id = session-xxx (per page load)

    API->>SVC: for event in run_turn(...)

    SVC->>Mem: append HumanMessage
    SVC->>R: classify question

    alt route = chat
        SVC->>LLM: stream(system + history)
    else route = ipl_narrative / ipl_stats / web_search
        SVC->>Tool: one tool call
        Tool-->>SVC: observation text
        SVC->>Mem: AIMessage + ToolMessage
        SVC->>LLM: stream(system + history + tool result)
    else route = browser_task / fallback
        SVC->>Agent: stream(messages)
        Agent->>LLM: ReAct (tools + answer)
        Agent->>Tool: optional multiple tools
    end

    loop Each AgentEvent
        SVC-->>API: thought_action | observation | answer
        API-->>HTML: SSE data: {JSON}\n\n
        HTML-->>You: Update steps + streaming answer
    end

    SVC->>Mem: append final AIMessage
```

### 3. How the UI talks to Python (contract)

```mermaid
flowchart TB
    subgraph Browser["Browser (same origin :8000)"]
        PAGE["index.html"]
        FETCH["fetch('/chat', POST)"]
        SSE["Read response body as SSE stream"]
        PAGE --> FETCH --> SSE
    end

    subgraph FastAPI["app/server.py"]
        ROOT["GET / → index.html"]
        CHAT["POST /chat → StreamingResponse"]
        CHAT --> GEN["json.dumps(AgentEvent) per line\ndata: ...\\n\\n"]
    end

    subgraph Events["AgentEvent types"]
        TA["thought_action — tool name/args"]
        OBS["observation — tool output"]
        ANS["answer — token chunk for reply"]
    end

    FETCH --> CHAT
    SSE --> Events
```

| `AgentEvent.type` | What you see in the UI |
|-------------------|-------------------------|
| `thought_action` | “Tool” step (e.g. `search_docs(...)`) |
| `observation` | Tool result snippet |
| `answer` | Assistant text streamed into the bubble |

Code anchors: `app/server.py` (`ChatRequest`, `/chat`), `static/index.html` (parse `data: ` lines), `app/service.py` (`run_turn` yields events).

### 4. Conversation context (short-term memory)

```mermaid
flowchart LR
    T1["Message 1"] --> STORE["_THREAD_STORES\nin RAM per thread_id"]
    T2["Message 2"] --> STORE
    STORE --> HIST["Full message list\nHuman / AI / Tool"]
    HIST --> LLM["Next turn:\nOllama sees prior turns"]

    REFRESH["Refresh browser page"] --> NEW["New thread_id\nhistory empty"]
    RESTART["Restart uvicorn"] --> CLEAR["All threads cleared"]
```

- **Same tab, same session:** follow-up questions can use earlier messages in that `thread_id`.
- **Not persisted:** restarting the server or refreshing the page starts a new thread (no disk DB yet; `app/memory.py` is Phase 5).

### 5. Where each question type goes (after routing)

```mermaid
flowchart TD
    Q["User question"] --> ROUTE["router.route()"]
    ROUTE -->|chat| CH["Direct LLM — no tools"]
    ROUTE -->|ipl_narrative| RAG["search_docs → Qdrant + embeddings"]
    ROUTE -->|ipl_stats| STATS["query_ipl_stats → pandas on CSV"]
    ROUTE -->|web_search| WEB["web_search → Tavily API"]
    ROUTE -->|browser_task| REACT["Full ReAct agent — all tools"]

    CH --> OUT["Stream answer to UI"]
    RAG --> OUT
    STATS --> OUT
    WEB --> OUT
    REACT --> OUT
```

---

## Master overview

```mermaid
flowchart TB
    subgraph Clients["User interfaces"]
        CLI["main.py — terminal chat"]
        WEB["static/index.html — browser UI"]
        VOICE_FUTURE["app/voice.py — future STT/TTS loop"]
    end

    subgraph API["Python API layer"]
        FAST["app/server.py — FastAPI POST /chat SSE"]
        SVC["app/service.py — run_turn()"]
        ROUTER["app/router.py — route: chat | ipl_* | web | browser"]
    end

    subgraph Agent["Agent core"]
        AGENT["app/agent.py — LangGraph ReAct + ChatOllama"]
        TOOLS["app/tools.py — tools"]
    end

    subgraph Local["On your machine (host)"]
        OLLAMA_LLM["Ollama — qwen3:8b :11434"]
        OLLAMA_EMB["Ollama — nomic-embed-text :11434"]
        PY["Python app + pandas CSVs data/"]
    end

    subgraph Docker["docker-compose"]
        QDRANT["Qdrant :6333"]
        VOL["./qdrant_data volume"]
    end

    subgraph External["Optional cloud"]
        TAVILY["Tavily API — web_search"]
        JEV["TypeSafe/Jev — routing"]
    end

    CLI --> SVC
    WEB --> FAST --> SVC
    VOICE_FUTURE -.-> SVC

    SVC --> ROUTER
    ROUTER --> JEV
    SVC --> AGENT
    SVC --> OLLAMA_LLM
    AGENT --> OLLAMA_LLM
    AGENT --> TOOLS

    TOOLS -->|search_docs| RAG["app/rag.py search()"]
    RAG --> OLLAMA_EMB
    RAG --> QDRANT
    QDRANT --- VOL

    TOOLS -->|query_ipl_stats| ANALYTICS["app/analytics.py"]
    ANALYTICS --> OLLAMA_LLM
    ANALYTICS --> PY

    TOOLS -->|web_search| TAVILY
```

**Talking point:** Almost everything is **local** (Ollama + Python + Qdrant in Docker). **Tavily** and **Jev routing** are the main optional external calls.

---

## 1. Python → chatbot → Ollama

Two entry paths share the same brain: `run_turn()` in `app/service.py`.

```mermaid
sequenceDiagram
    participant User
    participant UI as CLI or Browser
    participant API as FastAPI app/server.py
    participant SVC as service.run_turn
    participant Router as router.route
    participant LLM as ChatOllama → Ollama qwen3:8b
    participant Agent as LangGraph ReAct agent

    User->>UI: Type question
    alt Browser chat
        UI->>API: POST /chat {text, thread_id, mode}
        API->>SVC: run_turn() — SSE stream
    else Terminal
        UI->>SVC: run_turn() directly
    end

    SVC->>Router: Classify query (Jev / TypeSafe)
    alt chat / ipl_* / web_search
        SVC->>LLM: Direct stream OR tool then stream
    else browser_task / fallback
        SVC->>Agent: agent.stream(messages)
        Agent->>LLM: ReAct loop — think, tools, answer
        LLM-->>Agent: tool calls or tokens
    end

    SVC-->>UI: AgentEvents — thought_action, observation, answer
    UI-->>User: Show steps + final reply
```

| Layer | Role |
|--------|------|
| `main.py` | CLI; checks Ollama; `--mode text\|voice`, `--thread` |
| `app/server.py` | Serves UI + **Server-Sent Events** from `run_turn()` |
| `app/service.py` | Thread memory, routing, streaming events |
| `app/agent.py` | `get_ollama_llm()` → `http://localhost:11434`, `build_agent()` |

**Demo:** `python main.py` or Uvicorn on `app.server:app`.

---

## 2. How RAG connects (offline ingest vs online query)

RAG is **vectors in Qdrant** + **retrieval at tool time**, not weights inside the LLM.

```mermaid
flowchart LR
    subgraph Offline["One-time setup — NOT per chat message"]
        CSV["data/matches.csv + deliveries.csv"]
        BUILD["rag.build_match_documents()"]
        INGEST["rag.ingest()"]
        CSV --> BUILD --> INGEST
    end

    subgraph AtChat["Every narrative IPL question"]
        Q["User: Tell me about 2016 final"]
        R["router → ipl_narrative"]
        T["tools.search_docs(query)"]
        S["rag.search(query, k=4)"]
        E["Embed query: search_query: ... via Ollama"]
        QDR["Qdrant similarity_search"]
        CTX["XML retrieved_document chunks → LLM"]
        ANS["Ollama streams final answer"]

        Q --> R --> T --> S --> E --> QDR --> CTX --> ANS
    end

    INGEST --> QDRANT[(Qdrant collection ipl_matches)]
    QDR --> QDRANT
```

| Step | Code |
|------|------|
| Ingest CLI | `python -m scripts.ingest` → `app.rag.ingest()` |
| Retrieval tool | `search_docs` in `app/tools.py` → `app.rag.search()` |
| Qdrant URL | `QDRANT_URL=http://localhost:6333` (matches `docker-compose`) |
| Embeddings | `NomicEmbeddingsWrapper` — `search_document:` / `search_query:` prefixes |

**Parallel path (not vector RAG):** `ipl_stats` → `query_ipl_stats` → `app/analytics.py` (pandas on CSVs).

---

## 3. Chunking → Qdrant (Docker Compose)

```mermaid
flowchart TB
    subgraph Host["Host machine"]
        SCRIPT["scripts/ingest.py"]
        RAG["app/rag.py"]

        subgraph Build["build_match_documents()"]
            JOIN["Join matches + deliveries per match_id"]
            NARR["One narrative paragraph per match"]
            SPLIT["TokenTextSplitter chunk_size=500 overlap=50"]
            META["metadata: match_id, teams, season, venue, date, chunk_index"]
            JOIN --> NARR --> SPLIT --> META
        end

        subgraph Index["ingest()"]
            EMB["Ollama nomic-embed-text — search_document: text"]
            BATCH["add_documents batches of 50"]
            ID["uuid5 IDs: match_{id}_chunk_{n}"]
            META --> EMB --> BATCH --> ID
        end

        SCRIPT --> RAG
        RAG --> Build
        RAG --> Index
    end

    subgraph Compose["docker compose up -d"]
        Q["qdrant/qdrant:latest"]
        P["6333:6333 HTTP"]
        V["./qdrant_data → /qdrant/storage"]
        Q --> P
        Q --> V
    end

    ID -->|HTTP QDRANT_URL| Q
```

**Session bullets:**

- **Chunking** happens only during **ingest**, in `build_match_documents()` (`TokenTextSplitter`).
- Most match summaries are short → often **1 chunk per match**; splitter is a safety net for long text.
- **Persistence:** vectors live in `./qdrant_data` on the host; restarting the container keeps data.
- **Re-index:** `python -m scripts.ingest --reset` drops collection and rebuilds.

---

## 4. Tavily search

```mermaid
sequenceDiagram
    participant User
    participant SVC as run_turn
    participant Router as route → web_search
    participant Tool as tools.web_search
    participant Env as .env TAVILY_API_KEY
    participant Tavily as Tavily API HTTPS
    participant LLM as Ollama

    User->>SVC: "Latest IPL news today"
    SVC->>Router: Classify
    Router-->>SVC: web_search
    SVC->>Tool: web_search(query=user text)
    Tool->>Env: Read API key
    Tool->>Tavily: search max_results=3 depth=basic
    Tavily-->>Tool: title, url, content snippets
    Tool-->>SVC: Formatted observation
    SVC->>LLM: System prompt + history + tool result
    LLM-->>User: Grounded answer (streamed)
```

| Tool | Data source | Example |
|------|-------------|---------|
| `search_docs` | Qdrant + IPL narratives | "What happened in the 2016 final?" |
| `query_ipl_stats` | Local CSV + pandas | "Most runs in IPL history?" |
| `web_search` | **Tavily** (internet) | "Today's cricket headlines" |

If `TAVILY_API_KEY` is missing, the tool returns a clear error (search disabled).

**ReAct fallback:** `build_agent()` also registers `web_search` for full agent tool use.

---

## 5. Voice-ready Agent Lab

Short plain answers + browser STT/TTS; same backend as text.

```mermaid
flowchart TB
    subgraph Browser["static/index.html"]
        MIC["Web Speech API — SpeechRecognition"]
        SPEAKER["speechSynthesis — TTS"]
        MODE["mode: voice if mic or speaker on"]
        MIC -->|transcript| CHAT["POST /chat"]
        CHAT --> SPEAKER
    end

    subgraph Python["Same backend as text"]
        SVC["run_turn(..., mode='voice')"]
        PROMPT["VOICE_SYSTEM_PROMPT — 1–2 sentences, no markdown"]
        LLM["ChatOllama reasoning=false — faster"]
        SVC --> PROMPT --> LLM
    end

    subgraph Future["Planned"]
        VPY["app/voice.py — STT → run_turn → TTS"]
    end

    CHAT --> SVC
    VPY -.-> SVC
```

| Piece | Status |
|--------|--------|
| `mode=voice` + `VOICE_SYSTEM_PROMPT` | `app/agent.py` / `app/service.py` |
| CLI | `python main.py --mode voice` |
| Web UI | Mic + speaker; `mode: 'voice'` |
| `app/voice.py` | Stub — future STT/TTS loop |

---

## Suggested demo order

1. `docker compose up -d` → Qdrant on `:6333`
2. `python -m scripts.ingest` → chunking + embeddings (one-time)
3. `python main.py` or web UI → narrative → `search_docs` → Qdrant
4. Stats question → `query_ipl_stats`
5. Live news → `web_search` → Tavily
6. Voice / `--mode voice` → short answers

---

## ASCII deployment (optional slide)

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (index.html)  or  main.py CLI                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ run_turn(thread_id, mode)
┌───────────────────────────▼─────────────────────────────────┐
│  service.py → router → tools OR LangGraph ReAct agent        │
└─────┬──────────────┬──────────────┬─────────────────────────┘
      │              │              │
      ▼              ▼              ▼
 Ollama:11434    Qdrant:6333     Tavily API
 (qwen3 +        (docker-compose  (web_search)
  nomic-embed)    + qdrant_data)
      ▲
      │ ingest: CSV → chunk → embed → index
 scripts/ingest.py
```


flowchart LR
  User[User CLI / future Voice / API]
  Service[app/service.py run_turn]
  Agent[LangGraph ReAct agent]
  LLM[Ollama qwen3:8b]
  Tools[Tools layer]
  RAG[search_docs → Qdrant + embeddings]
  Stats[query_ipl_stats → pandas sandbox]
  Web[web_search → Tavily]
  User --> Service --> Agent --> LLM
  Agent --> Tools
  Tools --> RAG
  Tools --> Stats
  Tools --> Web
  RAG --> Qdrant[(Qdrant Docker)]
  RAG --> Embed[Ollama nomic-embed-text]
