"""LangGraph workshop — single-file browser UI for teaching.

Lessons (dropdown):
  1) Ollama only (no graph)
  2) Linear RAG graph (retrieve → answer)
  3) Memory (MemorySaver + thread_id)
  4) ReAct loop with RAG tool
  5) ReAct loop + Tavily web search

Run from project root:
  python workshop/langgraph_workshop_demo.py

Qdrant (default for this repo): start Docker + ingest first.
Chroma (classroom): set VECTOR_BACKEND=chroma and CHROMA_* in .env.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

# Allow `python workshop/langgraph_workshop_demo.py` from repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in ("1", "true", "yes")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant").lower()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cricket_matches")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Bump when you change workshop behavior (shown in UI — restart demo after git pull).
WORKSHOP_BUILD = "2026-03-19-trace-turn-fix"

LESSONS = [
    "1 — Ollama only (no LangGraph)",
    "2 — LangGraph: RAG (retrieve → answer)",
    "3 — LangGraph: memory (thread_id)",
    "4 — LangGraph: loop (RAG tool + ReAct)",
    "5 — LangGraph: loop + Tavily web search",
]


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _expand_retrieval_query(query: str) -> str:
    """Bias retrieval for common workshop questions.

    IPL ingest narratives describe matches by date/teams, not words like "final".
    Pure semantic search on "2016 IPL final" often misses match 636 (SRH vs RCB).
    """
    ql = query.lower()
    if "final" in ql and "2016" in ql:
        return (
            f"{query} IPL 2016 Sunrisers Hyderabad Royal Challengers Bangalore "
            "2016-05-29 M Chinnaswamy Stadium"
        )
    return query


IPL_2016_FINAL_MATCH_ID = 636


def _inject_2016_final_hit(query: str, hits: list[dict]) -> tuple[list[dict], bool]:
    """If vector search omits match 636, load it directly from Qdrant by ID."""
    ql = query.lower()
    if "final" not in ql or "2016" not in ql:
        return hits, False
    if any(
        (h.get("metadata") or {}).get("match_id") == IPL_2016_FINAL_MATCH_ID
        for h in hits
    ):
        return hits, False
    from app.rag import get_match_by_id

    direct = get_match_by_id(IPL_2016_FINAL_MATCH_ID)
    if direct:
        return [direct, *hits], True
    return hits, False


def _rerank_hits(query: str, hits: list[dict]) -> list[dict]:
    """Put the true 2016 final (match 636) first when the user asks about that final."""
    ql = query.lower()
    if "final" not in ql or "2016" not in ql:
        return hits

    def priority(h: dict) -> tuple[int, float]:
        meta = h.get("metadata") or {}
        mid = meta.get("match_id")
        text = h.get("text") or ""
        boost = 0
        if mid in (IPL_2016_FINAL_MATCH_ID, str(IPL_2016_FINAL_MATCH_ID)):
            boost = 100
        elif "2016-05-29" in text:
            boost = 50
        score = float(h.get("score") or 0)
        return (boost, score)

    return sorted(hits, key=priority, reverse=True)


def _vector_search(query: str, k: int = 4) -> tuple[str, list[int], str]:
    """Return formatted context, match_ids, and extra trace notes."""
    search_query = _expand_retrieval_query(query)
    extra_trace = ""
    if VECTOR_BACKEND == "qdrant":
        from app.rag import search

        hits = search(search_query, k=max(k, 12))
        if not hits:
            return (
                "(No Qdrant results — run `docker compose up -d` and "
                "`python -m scripts.ingest --reset`.)",
                [],
                extra_trace,
            )
        hits, injected = _inject_2016_final_hit(query, hits)
        if injected:
            extra_trace = (
                " | direct_fetch match_id=636 (vector search missed the final)"
            )
        hits = _rerank_hits(query, hits)
        hits = hits[:k]
        match_ids = [
            int((h.get("metadata") or {}).get("match_id") or 0) for h in hits
        ]
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata") or {}
            parts.append(f"[{i}] {h.get('text', '')}\nmeta: {meta}")
        return "\n\n".join(parts), match_ids, extra_trace

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    docs = store.similarity_search(search_query, k=k)
    if not docs:
        return (
            "(No Chroma documents — check CHROMA_PATH / CHROMA_COLLECTION.)",
            [],
            extra_trace,
        )
    parts = []
    match_ids: list[int] = []
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        mid = meta.get("match_id")
        if mid is not None:
            match_ids.append(int(mid))
        parts.append(f"[{i}] {doc.page_content}\nmeta: {meta}")
    return "\n\n".join(parts), match_ids, extra_trace


@tool
def search_cricket_docs(query: str) -> str:
    """Search stored cricket match narratives (RAG). Use for IPL/historical match questions."""
    context, _, _ = _vector_search(query, k=4)
    return context


@tool
def web_search_tavily(query: str) -> str:
    """Search the live web for recent news. Use for today's scores, breaking news, weather."""
    if not TAVILY_API_KEY:
        return "Tavily disabled: set TAVILY_API_KEY in .env"
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=query, max_results=3, search_depth="basic")
        lines = []
        for r in res.get("results", []):
            title = r.get("title", "")
            content = (r.get("content") or "")[:400]
            lines.append(f"- {title}: {content}")
        return "\n".join(lines) if lines else "No results."
    except Exception as exc:
        return f"Tavily error: {exc}"


def make_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=ENABLE_THINKING,
    )


class RagState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    trace: str
    match_ids: list[int]


def _node_retrieve(state: RagState) -> dict:
    last = state["messages"][-1]
    q = last.content if isinstance(last, HumanMessage) else str(last)
    expanded = _expand_retrieval_query(q)
    ctx, match_ids, extra = _vector_search(q, k=6)
    trace = "nodes: retrieve"
    if expanded != q:
        trace += " (query expanded for '2016 final' — narratives omit the word 'final')"
    if extra:
        trace += extra
    if match_ids:
        trace += f" | match_ids={match_ids}"
    return {"context": ctx, "trace": trace, "match_ids": match_ids}


def _node_rag_answer(state: RagState, temperature: float) -> dict:
    llm = make_llm(temperature)
    last = state["messages"][-1]
    user = last.content if isinstance(last, HumanMessage) else str(last)
    ql = (user or "").lower()

    ctx = (state.get("context") or "").strip()
    match_ids = list(state.get("match_ids") or [])
    # Safety: always have retrieval text at generate time (avoids empty context edge cases).
    if not ctx or ctx.startswith("(No ") or len(ctx) < 80:
        ctx, match_ids, _extra = _vector_search(user, k=6)

    if "final" in ql and "2016" in ql and match_ids and match_ids[0] == 636:
        # Top chunk after rerank is the final — keeps the LLM from fixating on "final" wording.
        if "\n\n[2]" in ctx:
            ctx = ctx.split("\n\n[2]", 1)[0]

    final_hint = ""
    if "final" in ql and "2016" in ql:
        final_hint = (
            "\nThe user is asking about the 2016 IPL final. In this dataset that is "
            "match 636 (2016-05-29): Sunrisers Hyderabad vs Royal Challengers Bangalore "
            "at M Chinnaswamy Stadium. Summarize that match from CONTEXT even though "
            "the word 'final' may not appear in the text.\n"
        )
    system = (
        "You are a cricket assistant. Answer using the CONTEXT below. "
        "The context lists IPL match summaries (teams, date, scores, winner). "
        "Pick the summary that best answers the question."
        f"{final_hint}"
        "If CONTEXT says no Qdrant/Chroma results or is empty, say you don't know.\n\n"
        f"CONTEXT:\n{ctx}"
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    trace = state.get("trace", "") or "nodes: retrieve"
    if match_ids and "match_ids=" not in trace:
        trace += f" | match_ids={match_ids}"
    trace += f" | context_chars={len(ctx)} → generate"
    return {
        "messages": [AIMessage(content=text)],
        "trace": trace,
        "context": ctx,
        "match_ids": match_ids,
    }


def build_rag_graph(temperature: float):
    graph = StateGraph(RagState)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("generate", lambda s: _node_rag_answer(s, temperature))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    trace: str


_memory = MemorySaver()


def build_react_graph(temperature: float, use_tavily: bool):
    tools = [search_cricket_docs]
    if use_tavily:
        tools.append(web_search_tavily)

    llm = make_llm(temperature).bind_tools(tools)

    def agent_node(state: ChatState) -> dict:
        out = llm.invoke(state["messages"])
        return {"messages": [out]}

    graph = StateGraph(ChatState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=_memory)


def _history_to_messages(history: list) -> list:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def _messages_for_current_turn(all_messages: list, user_message: str) -> list:
    """With MemorySaver, graph state includes the whole thread — trace only this turn."""
    start = 0
    for i, m in enumerate(all_messages):
        if isinstance(m, HumanMessage) and (m.content or "") == user_message:
            start = i
    return all_messages[start:]


def run_chat(
    message: str,
    history: list,
    lesson: str,
    temperature: float,
    thread_id: str,
):
    if not message.strip():
        return history, "", ""

    trace_lines = [
        f"lesson={lesson}",
        f"temperature={temperature}",
        f"thread_id={thread_id}",
        f"vectors={VECTOR_BACKEND}",
    ]

    if lesson.startswith("1"):
        llm = make_llm(temperature)
        msgs = _history_to_messages(history)
        msgs.append(HumanMessage(content=message))
        resp = llm.invoke(msgs)
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)
        trace_lines.append("path: ChatOllama.invoke (no LangGraph)")
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, "", "\n".join(trace_lines)

    if lesson.startswith("2"):
        graph = build_rag_graph(temperature)
        trace_lines.append("(thread_id is ignored in lesson 2 — not used until lesson 3)")
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "context": "",
                "trace": "",
                "match_ids": [],
            }
        )
        answer = result["messages"][-1].content
        trace_lines.append(result.get("trace", ""))
        ctx = result.get("context") or ""
        if ctx.startswith("(No Qdrant") or ctx.startswith("(No Chroma"):
            trace_lines.append(
                "FIX: docker compose up -d && python -m scripts.ingest --reset"
            )
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, "", "\n".join(trace_lines)

    use_tavily = lesson.startswith("5")
    graph = build_react_graph(temperature, use_tavily=use_tavily)
    config = {"configurable": {"thread_id": thread_id}}

    if lesson.startswith("3"):
        system = SystemMessage(
            content=(
                "You are a friendly cricket chatbot. Remember what the user "
                "told you earlier in this thread."
            )
        )
    else:
        system = SystemMessage(
            content=(
                "Cricket assistant with tools. "
                "Use search_cricket_docs for match history and narratives. "
                + (
                    "Use web_search_tavily for live or recent web facts. "
                    if use_tavily
                    else ""
                )
                + "Call tools when needed, then answer clearly."
            )
        )

    result = graph.invoke(
        {"messages": [system, HumanMessage(content=message)]},
        config=config,
    )

    turn_messages = _messages_for_current_turn(result["messages"], message)
    tool_events = 0
    for m in turn_messages:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                trace_lines.append(f"tool_call: {tc['name']}({tc['args']})")
                tool_events += 1
        if isinstance(m, ToolMessage):
            preview = str(m.content)[:200]
            trace_lines.append(f"tool_result: {m.name} → {preview}...")
            tool_events += 1

    if tool_events:
        trace_lines.append("loop: agent ⇄ tools until END (this turn only)")
    else:
        trace_lines.append("no tools called this turn (answered from model only)")

    answer_msg = result["messages"][-1]
    answer = (
        answer_msg.content
        if isinstance(answer_msg.content, str)
        else str(answer_msg.content)
    )
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, "", "\n".join(trace_lines)


def build_ui():
    with gr.Blocks(title="LangGraph Workshop") as demo:
        gr.Markdown(
            f"**Workshop build:** `{WORKSHOP_BUILD}` (restart this script after code updates)  \n"
            f"**Ollama:** `{OLLAMA_MODEL}` @ `{OLLAMA_BASE_URL}`  \n"
            f"**Vector store:** `{VECTOR_BACKEND}`"
        )
        with gr.Row():
            lesson = gr.Dropdown(LESSONS, value=LESSONS[0], label="Lesson")
            temperature = gr.Slider(
                0, 1.5, value=0.2, step=0.1, label="Temperature"
            )
            thread_id = gr.Textbox(
                value="workshop-thread-1",
                label="Thread ID (lessons 3–5 only; ignored in 1–2)",
            )
        chatbot = gr.Chatbot(height=420)
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask about a match, or use lesson 3 to test memory",
        )
        trace = gr.Textbox(label="Teaching trace (nodes / tools)", lines=8)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear chat")

        inputs = [msg, chatbot, lesson, temperature, thread_id]
        outputs = [chatbot, msg, trace]
        send.click(run_chat, inputs=inputs, outputs=outputs)
        msg.submit(run_chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: ([], ""), outputs=[chatbot, trace])

    return demo


if __name__ == "__main__":
    build_ui().launch()
