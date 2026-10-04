"""Agent Service Layer (Phase 1).

Why this file exists:
Decouples the core agent execution from any specific front-end (CLI, Voice, Web, API).
By isolating the ReAct turn execution into `run_turn()`, this service can be invoked by:
  1. main.py (interactive terminal CLI)
  2. app/voice.py (future voice loop with Speech-To-Text and Text-To-Speech)
  3. A FastAPI / WebSocket server for browser or mobile frontends

Conversation State Design:
All conversation state is partitioned by `thread_id`. In Phase 1, state is stored in an
in-memory thread dictionary, and requests pass `config={"configurable": {"thread_id": thread_id}}`.
In Phase 5, a SQLite or Postgres checkpointer will plug directly into this contract
without requiring any refactoring of callers.
"""

import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Literal
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.state import CompiledStateGraph

from app.agent import (
    TEXT_SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT,
    build_agent,
    get_ollama_llm,
    strip_thinking,
)
from app.router import route
from app.tools import query_ipl_stats, search_docs, web_search



@dataclass
class AgentEvent:
    """Represents a discrete step in the agent's ReAct execution cycle.

    Attributes:
        type: The phase of the step:
            - 'thought_action': Model reasoning and tool call request.
            - 'observation': Result returned from tool execution.
            - 'answer': Final conversational answer produced by the model.
        content: Text content (reasoning thoughts, observation text, or final answer).
        tool_name: Name of the invoked tool (if applicable).
        tool_args: Arguments dictionary passed to the tool (if applicable).
    """

    type: Literal["thought_action", "observation", "answer"]
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


# In-memory thread storage for Phase 1.
# Key: thread_id (str), Value: List[BaseMessage]
# Phase 5 will seamlessly replace/augment this with LangGraph Checkpointer (e.g. SQLite).
_THREAD_STORES: dict[str, list[BaseMessage]] = defaultdict(list)

# Cache compiled agent graphs by mode ('text' vs 'voice')
_COMPILED_AGENTS: dict[str, CompiledStateGraph] = {}


# Tools mapped to their input argument name for single-tool execution
_ROUTE_TOOLS = {
    "ipl_narrative": (search_docs, "query"),
    "ipl_stats": (query_ipl_stats, "question"),
    "web_search": (web_search, "query"),
}


def _get_direct_llm_and_prompt(mode: str = "text") -> tuple[Any, str]:
    """Retrieve chat LLM and appropriate system prompt for non-agent turns."""
    thinking_enabled = (
        False
        if mode == "voice"
        else os.getenv("ENABLE_THINKING", "false").lower() in ("true", "1", "yes")
    )
    prompt = VOICE_SYSTEM_PROMPT if mode == "voice" else TEXT_SYSTEM_PROMPT
    llm = get_ollama_llm(enable_thinking=thinking_enabled)
    return llm, prompt


def get_agent(mode: str = "text") -> CompiledStateGraph:
    """Retrieve or compile the agent graph for the given mode."""
    if mode not in _COMPILED_AGENTS:
        _COMPILED_AGENTS[mode] = build_agent(mode=mode)
    return _COMPILED_AGENTS[mode]


def get_thread_history(thread_id: str = "default") -> list[BaseMessage]:
    """Retrieve the conversation message history for a given thread."""
    return list(_THREAD_STORES[thread_id])


def clear_thread(thread_id: str = "default") -> None:
    """Clear conversation history for a given thread."""
    if thread_id in _THREAD_STORES:
        _THREAD_STORES[thread_id].clear()


def run_turn(
    user_text: str,
    thread_id: str = "default",
    mode: str = "text",
) -> Iterator[AgentEvent]:
    """Execute a single conversational turn through the ReAct agent graph.

    Args:
        user_text: The user's input text (from CLI, voice STT, or API).
        thread_id: Identifier for the conversation session (defaults to 'default').
        mode: 'text' (default, detailed markdown) or 'voice' (short, plain conversational).

    Yields:
        AgentEvent: Step-by-step events:
            - 'thought_action': When model decides to call tools.
            - 'observation': When tool returns execution data.
            - 'answer': When model produces the final synthesized response.
    """
    clean_input = user_text.strip()
    if not clean_input:
        return

    # Add user message to conversation history for this thread
    user_message = HumanMessage(content=clean_input)
    thread_history = _THREAD_STORES[thread_id]
    initial_len = len(thread_history)
    thread_history.append(user_message)

    # 1. Route the query using TypeSafe System One (Jev). Fall back to ReAct agent if Jev fails.
    try:
        route_decision = route(clean_input)
    except Exception:
        route_decision = "browser_task"

    try:
        # Route: chat -> one direct LLM call, no tools; stream tokens
        if route_decision == "chat":
            llm, system_prompt = _get_direct_llm_and_prompt(mode=mode)
            accumulated = ""
            for token in llm.stream([SystemMessage(content=system_prompt)] + thread_history):
                text = strip_thinking(token.content)
                if text:
                    accumulated += text
                    yield AgentEvent(type="answer", content=text)
            thread_history.append(AIMessage(content=accumulated))
            return

        # Route: ipl_narrative / ipl_stats / web_search -> single tool, then stream LLM answer
        if route_decision in _ROUTE_TOOLS:
            tool, arg_name = _ROUTE_TOOLS[route_decision]
            tool_args = {arg_name: clean_input}
            yield AgentEvent(
                type="thought_action",
                content="",
                tool_name=tool.name,
                tool_args=tool_args,
            )
            tool_result = str(tool.invoke(tool_args))
            yield AgentEvent(
                type="observation",
                content=tool_result,
                tool_name=tool.name,
            )

            call_id = f"call_{uuid.uuid4().hex[:8]}"
            thread_history.append(
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": tool.name, "args": tool_args}],
                )
            )
            thread_history.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=call_id,
                    name=tool.name,
                )
            )

            llm, system_prompt = _get_direct_llm_and_prompt(mode=mode)
            accumulated = ""
            for token in llm.stream([SystemMessage(content=system_prompt)] + thread_history):
                text = strip_thinking(token.content)
                if text:
                    accumulated += text
                    yield AgentEvent(type="answer", content=text)
            thread_history.append(AIMessage(content=accumulated))
            return

        # Route: browser_task or fallback -> existing ReAct agent with token streaming
        agent = get_agent(mode=mode)
        graph_config = {"configurable": {"thread_id": thread_id}}

        for stream_type, data in agent.stream(
            {"messages": thread_history},
            config=graph_config,
            stream_mode=["updates", "messages"],
        ):
            if stream_type == "updates":
                # "updates" carries completed messages for history tracking and tool events
                for node_name, update in data.items():
                    for msg in update.get("messages", []):
                        thread_history.append(msg)
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield AgentEvent(
                                    type="thought_action",
                                    content=msg.content or "",
                                    tool_name=tc.get("name"),
                                    tool_args=tc.get("args"),
                                )
                        elif isinstance(msg, ToolMessage):
                            yield AgentEvent(
                                type="observation",
                                content=str(msg.content),
                                tool_name=getattr(msg, "name", "tool"),
                            )
            elif stream_type == "messages":
                # "messages" carries per-token AIMessageChunks for live answer streaming
                msg, _meta = data
                if (
                    isinstance(msg, AIMessageChunk)
                    and msg.content
                    and not getattr(msg, "tool_call_chunks", None)
                ):
                    text = strip_thinking(msg.content)
                    if text:
                        yield AgentEvent(type="answer", content=text)

    except Exception:
        # Rollback all messages added during this turn if it fails
        while len(thread_history) > initial_len:
            thread_history.pop()
        raise

