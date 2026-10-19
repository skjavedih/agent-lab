# How to test the LangGraph workshop demo

Step-by-step checklist for `langgraph_workshop_demo.py`. Use the **Teaching trace** panel in the UI to confirm each lesson ran the expected path.

**Run the demo:**

```bash
source .venv/bin/activate
python workshop/langgraph_workshop_demo.py
```

Open the URL Gradio prints (usually http://127.0.0.1:7860).

---

## Before you start (once per machine)

1. **Ollama** running; models pulled:

   ```bash
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```

2. **RAG data** (default `VECTOR_BACKEND=qdrant` in this repo):

   ```bash
   docker compose up -d
   python -m scripts.ingest --reset
   ```

   Classroom **Chroma** instead: set `VECTOR_BACKEND=chroma` and `CHROMA_PATH` / `CHROMA_COLLECTION` in `.env`.

3. **Lesson 5 only:** set `TAVILY_API_KEY` in `.env` and restart the demo.

**Tips:**

- Click **Clear chat** between lessons for a clean chat panel.
- Lessons **3–5** store memory in LangGraph keyed by **Thread ID**. Clearing the UI does **not** reset that memory. Use a new thread (e.g. `workshop-thread-2`) or restart Python for a fresh memory demo.

---

## Lesson 1 — Ollama only (no LangGraph)

**Goal:** LLM only — no graph, no RAG (same idea as a simple Ollama chatbot).

| Step | Action | Pass if |
|------|--------|---------|
| 1 | Select **1 — Ollama only** | Header shows model and base URL |
| 2 | Temperature **0.2** | — |
| 3 | Ask: *What is LBW in cricket?* | Sensible general answer |
| 4 | Read **Teaching trace** | Contains `path: ChatOllama.invoke (no LangGraph)` |
| 5 | Ask: *Who won the 2016 IPL final?* | May guess from model weights — **not** from Qdrant |
| 6 | Set temperature **1.2**, repeat the LBW question | More varied wording |

**Talking point:** Without RAG, factual IPL answers can be wrong — that motivates lesson 2.

**Why lesson 1 can be right and lesson 2 says “I don’t know”:** Lesson 1 uses the model’s **built-in knowledge**. Lesson 2 must use **retrieved text only**. Our match documents say things like “Match 636 … 2016-05-29 … Sunrisers Hyderabad vs Royal Challengers Bangalore” but often **never use the word “final”**, so a query like “2016 IPL final” can retrieve the wrong 2016 games. The model then correctly refuses to invent an answer. The workshop expands that query before search; you can also ask with **teams + date** (see lesson 2 below).

---

## Lesson 2 — LangGraph: RAG (retrieve → answer)

**Goal:** Two-node graph: **retrieve** from vectors, then **generate** an answer.

| Step | Action | Pass if |
|------|--------|---------|
| 1 | **Clear chat**, select **lesson 2** | — |
| 2 | Temperature **0.2**; **Thread ID does not matter** (only lessons 3–5 use it) | — |
| 3 | Ask: *Tell me about the 2016 IPL final between Sunrisers Hyderabad and Royal Challengers Bangalore.* | Answer matches ingested match narratives |
| 4 | Read **Teaching trace** | `nodes: retrieve → generate` and `vectors=qdrant` (or `chroma`) |
| 5 | Ask something off-topic: *Who won the 2016 FIFA World Cup in IPL terms?* | Weak or “don’t know” (context-only prompt) |

**Match 636 not in `inspect_retrieval`?** That script uses plain vector search. League games (617, 580, …) often rank above the real final because old ingested text never said “final”. The workshop fixes this by **direct_fetch match_id=636** when needed (see trace). Check the doc exists:

```bash
python -c "from app.rag import get_match_by_id; print(get_match_by_id(636)['metadata'])"
```

Re-ingest after pulling latest `app/rag.py` so new narratives include “IPL 2016 final” wording:

```bash
python -m scripts.ingest --reset
```

**If you still get “don’t know” on the final:** trace should show `match_ids=[636, ...]` and `context_chars` > 500. If you see `(No Qdrant results)`, run Docker + ingest.

**Alternate phrasing that retrieves well without expansion:** *Sunrisers Hyderabad vs Royal Challengers Bangalore 29 May 2016*

**Note:** Each turn uses only the **current** message in the graph (not Gradio history). Good for teaching RAG, not long chat memory.

**Optional CLI check (no LLM):**

```bash
python -m scripts.inspect_retrieval "2016 IPL final"
```

---

## Lesson 3 — LangGraph: memory (`thread_id`)

**Goal:** `MemorySaver` + **Thread ID** — the graph remembers prior turns for that thread.

| Step | Action | Pass if |
|------|--------|---------|
| 1 | **Clear chat**, select **lesson 3** | — |
| 2 | Thread ID: `workshop-thread-1` | — |
| 3 | *My name is Priya and my favorite IPL team is Chennai Super Kings.* | Acknowledgment |
| 4 | *What is my name?* | **Priya** |
| 5 | *What is my favorite team?* | **CSK** / Chennai Super Kings |
| 6 | Read trace | `thread_id=workshop-thread-1`, `loop: agent ⇄ tools until END` (usually no tool calls) |
| 7 | Change Thread ID to `workshop-thread-2`, ask *What is my name?* | Does not know Priya |
| 8 | Switch back to `workshop-thread-1`, ask *What is my name?* | **Priya** again |

**Talking point:** Same app, different `thread_id` = different conversation state.

---

## Lesson 4 — LangGraph: loop (RAG tool + ReAct)

**Goal:** Model chooses **`search_cricket_docs`**, reads tool output, then answers.

| Step | Action | Pass if |
|------|--------|---------|
| 1 | **Clear chat**, select **lesson 4** | — |
| 2 | Thread ID: `workshop-thread-4` | — |
| 3 | Temperature **0.2** | — |
| 4 | Ask: *Describe the 2016 IPL final.* | Detailed narrative |
| 5 | Read **Teaching trace** | `tool_call: search_cricket_docs(...)`, then `tool_result: search_cricket_docs → ...`, then `loop: agent ⇄ tools until END` |
| 6 | Ask: *What is 15 × 23?* | Trace should say **`no tools called this turn`** — not cricket tool lines from an earlier question on the same thread (fixed in workshop build `2026-03-19-trace-turn-fix`) |
| 7 | Ask: *Use the match database: who won the 2016 IPL final?* | RAG tool in trace if needed |

**If RAG never appears:** check Docker, ingest, and `nomic-embed-text`.

---

## Lesson 5 — Loop + Tavily web search

**Goal:** **RAG** for local match data; **Tavily** for live web.

| Step | Action | Pass if |
|------|--------|---------|
| 1 | `TAVILY_API_KEY` set, demo restarted | — |
| 2 | **Clear chat**, select **lesson 5**, thread `workshop-thread-5` | — |
| 3 | *Summarize the 2016 IPL final from our data.* | `search_cricket_docs` in trace |
| 4 | *What are today’s top cricket headlines?* | `web_search_tavily` in trace |
| 5 | Remove key, restart, repeat step 4 | Tool result mentions Tavily disabled |

---

## Temperature demo (any lesson with generation)

Use lesson **2** or **4** with the same factual question twice:

| Temperature | Example question | What to show |
|-------------|------------------|--------------|
| **0.1** | *Who won the 2016 IPL final?* | Stable, factual tone |
| **1.2** | Same question | More varied phrasing; watch for drift |

Trace always includes `temperature=...`.

---

## Quick reference (on stage)

| Lesson | Trace should show | Good first question |
|--------|-------------------|---------------------|
| 1 | `ChatOllama.invoke (no LangGraph)` | What is LBW? |
| 2 | `retrieve → generate` | 2016 IPL final narrative |
| 3 | Same thread recalls facts | Name + team, then “What’s my name?” |
| 4 | `search_cricket_docs` | Describe 2016 IPL final |
| 5 | `web_search_tavily` | Today’s cricket news |

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Lesson 2/4 empty or “don’t know” | `docker compose up -d`, `python -m scripts.ingest --reset`, pull `nomic-embed-text` |
| Lesson 3 doesn’t remember | Use the same **Thread ID** across turns; lesson 1 history is not LangGraph memory |
| Lesson 5 no web | Set `TAVILY_API_KEY`, restart script |
| Slow responses | `ENABLE_THINKING=false` in `.env` |

---

See also: [README.md](../README.md) (setup step 8) and [Architecture-diagram.md](../Architecture-diagram.md).
