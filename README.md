# 🧪 Agent Lab

A modular AI agent in Python built with LangGraph and local Ollama models, featuring ReAct step streaming, Qdrant vector RAG over Kaggle IPL data, sandboxed LLM-assisted pandas analytics, Tavily live web search, multi-session threads, and voice-ready formatting.

---

## 📌 Architecture & Design

### Decoupled Service Layer (`app/service.py`)
The agent core is decoupled from the CLI. Frontends interact through a single generator function:
```python
from app.service import run_turn

for event in run_turn(user_text="Who won the 2016 IPL final?", thread_id="session-1", mode="text"):
    if event.type == "thought_action":
        print(f"Tool call: {event.tool_name}({event.tool_args})")
    elif event.type == "observation":
        print(f"Result: {event.content}")
    elif event.type == "answer":
        print(f"Final answer: {event.content}")
```
This enables `main.py` (CLI), `app/voice.py` (future Speech-To-Text / Text-To-Speech loop), or a FastAPI server to run the same agent without changes.

### Dual Query Paths (Phase 3: RAG + Analytics)
1. **Narrative Path (`app/rag.py` / `search_docs` tool)**:
   - Evaluates match-level questions ("tell me about...", "describe the 2016 final", "what happened when...").
   - Each IPL match in `matches.csv` is synthesized with `deliveries.csv` into a comprehensive natural-language paragraph: teams, date, venue, toss, innings totals, winner/margin, Player of the Match, top scorer, and best bowler.
   - Embeds with `nomic-embed-text` via Ollama using task-specific prefixes (`search_document: ` / `search_query: `) and stores in Qdrant with deterministic point IDs.
2. **Structured Path (`app/analytics.py` / `query_ipl_stats` tool)**:
   - Evaluates quantitative, aggregation, and ranking questions ("how many matches did X win", "most runs", "average score", "highest team total").
   - Translates questions into minimal pandas operations executed in a sandboxed namespace containing only `matches`, `deliveries`, and safe builtins (omitting `open`, `exec`, `eval`, `__import__`).

### Dual Operating Modes (`--mode text` | `--mode voice`)
- **`text` mode**: Comprehensive answers with markdown, tables, and code snippets.
- **`voice` mode**: Conversational responses designed for TTS engines:
  - 2 to 3 sentences maximum.
  - No markdown, no bold/italics, no bullet points.
  - No code blocks, no tables, no emojis.
  - Thinking tokens disabled for fast response times.

### Stateless-Friendly Threading (`--thread <id>`)
Conversation state is managed per `thread_id` and passed through LangGraph's `config={"configurable": {"thread_id": thread_id}}`. In Phase 5, a SQLite checkpointer can be plugged in directly with zero refactoring.

---

## 📊 Confirmed Dataset Schema (`data/`)

Verified via `scripts/inspect_data.py`:

### `matches.csv` (636 rows, 18 columns)
- **Columns**: `id` (int64), `season` (int64, 2008–2017), `city` (str), `date` (datetime/str), `team1` (str), `team2` (str), `toss_winner` (str), `toss_decision` (str: 'field'/'bat'), `result` (str: 'normal', 'tie', 'no result'), `dl_applied` (int64: 0/1), `winner` (str), `win_by_runs` (int64), `win_by_wickets` (int64), `player_of_match` (str), `venue` (str), `umpire1` (str), `umpire2` (str), `umpire3` (float64, 100% null).
- **Null Characteristics**:
  - `city`: 7 null values (all for Dubai International Cricket Stadium matches in 2014).
  - `winner`: 3 null values (all correspond to matches with `result == 'no result'`).
  - `player_of_match`: 3 null values (matches with no result).
  - `umpire3`: 636 nulls (100% NaN).

### `deliveries.csv` (150,460 rows, 21 columns)
- **Columns**: `match_id` (int64, joins to `matches.id`), `inning` (int64: 1–4), `batting_team` (str), `bowling_team` (str), `over` (int64: 1–20), `ball` (int64: 1–9), `batsman` (str), `non_striker` (str), `bowler` (str), `is_super_over` (int64: 0/1), `wide_runs` (int64), `bye_runs` (int64), `legbye_runs` (int64), `noball_runs` (int64), `penalty_runs` (int64), `batsman_runs` (int64), `extra_runs` (int64), `total_runs` (int64), `player_dismissed` (str), `dismissal_kind` (str), `fielder` (str).
- **Null Characteristics**:
  - `player_dismissed`, `dismissal_kind`: 143,022 nulls (~95.1%, only populated on wicket deliveries).
  - Bowler wicket dismissals count: `caught`, `bowled`, `lbw`, `stumped`, `caught and bowled`, `hit wicket` (excludes `run out`, `retired hurt`, `obstructing the field`).

---

## 🚀 Setup & Execution in Order

### 1. Prerequisites & Models
Ensure Ollama is running and both models are pulled:
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 2. Environment Configuration
Verify your `.env` file (copied from `.env.example`):
```env
# Tavily API key for live web search
TAVILY_API_KEY=your_key_here

# Local Ollama LLM configuration
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_THINKING=false

# Vector Database & Embeddings (Phase 3)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ipl_matches
EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Step-by-Step Commands

1. **Inspect Data**:
   ```bash
   python scripts/inspect_data.py
   ```
2. **Start Qdrant Container**:
   ```bash
   docker compose up -d
   ```
3. **Ingest Match Documents into Qdrant**:
   ```bash
   python -m scripts.ingest --reset
   ```
4. **Test Vector Retrieval (No LLM call)**:
   ```bash
   python -m scripts.inspect_retrieval "Who won the 2016 IPL final?"
   ```
5. **Test Structured Stats Query**:
   ```bash
   python -c "from app.analytics import query_ipl_stats; print(query_ipl_stats('How many matches has Mumbai Indians won in total?'))"
   ```
6. **Start Interactive Chat**:
   ```bash
   python main.py
   ```

---

## 🧪 Example Test Questions

### 📖 Narrative Questions (Routes to `search_docs`)
1. *"Tell me about the 2016 IPL final between Sunrisers Hyderabad and Royal Challengers Bangalore."*
2. *"Describe the match where Gujarat Lions played Mumbai Indians in Rajkot in 2017."*
3. *"Which match in 2014 was played in Dubai between Mumbai Indians and Royal Challengers Bangalore?"*

### 🔢 Quantitative / Statistical Questions (Routes to `query_ipl_stats`)
1. *"How many total matches has Mumbai Indians won across all seasons?"*
2. *"Who scored the most runs overall in IPL history?"*
3. *"Which player won the most Player of the Match awards?"*

---

## 📂 Project Structure

```text
agent-lab/
├── app/
│   ├── __init__.py
│   ├── agent.py               # ReAct agent definition & system prompt with routing guidance
│   ├── analytics.py           # Structured pandas query translation & sandboxed execution
│   ├── guardrails.py          # Phase 7 placeholder: Safety validation
│   ├── memory.py              # Phase 5 placeholder: SQLite checkpointer
│   ├── rag.py                 # Match document builder, Nomic embeddings wrapper, Qdrant store
│   ├── router.py              # Phase 8 placeholder: Semantic query router
│   ├── service.py             # Decoupled service layer: run_turn() generator & thread state
│   ├── tools.py               # Tools: search_docs, query_ipl_stats, calculator, time, search
│   └── voice.py               # Phase 2: Speech-To-Text / Text-To-Speech loop
├── data/
│   ├── deliveries.csv         # Kaggle IPL ball-by-ball dataset (150k rows)
│   └── matches.csv            # Kaggle IPL match-by-match dataset (636 rows)
├── scripts/
│   ├── ingest.py              # CLI to embed and ingest documents into Qdrant
│   ├── inspect_data.py        # Inspects CSV shape, dtypes, samples, and nulls
│   └── inspect_retrieval.py   # Test vector retrieval top-k with scores without LLM
├── docker-compose.yml         # Qdrant vector database container configuration
├── main.py                    # Interactive CLI wrapper supporting --mode and --thread
├── requirements.txt           # Dependencies (pandas, langchain, qdrant-client, ollama)
├── .env.example               # Environment variables template
├── .gitignore                 # Git exclusion rules
└── README.md                  # Comprehensive documentation
```
