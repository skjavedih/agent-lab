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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent import build_agent, strip_thinking


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

    # 1. Retrieve the appropriate agent for this mode ('text' or 'voice')
    agent = get_agent(mode=mode)

    # 2. Add user message to conversation history for this thread
    user_message = HumanMessage(content=clean_input)
    thread_history = _THREAD_STORES[thread_id]
    thread_history.append(user_message)

    # 3. LangGraph configuration with thread_id for state isolation
    # Passing thread_id here ensures full compatibility with Phase 5 checkpointers
    graph_config = {"configurable": {"thread_id": thread_id}}

    try:
        # Stream updates from the ReAct graph
        for chunk in agent.stream(
            {"messages": thread_history},
            config=graph_config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                messages = update.get("messages", [])
                for msg in messages:
                    # Append all intermediate steps (AIMessage, ToolMessage) to keep history consistent
                    thread_history.append(msg)

                    # Reason / Act step: The LLM decided to invoke one or more tools
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield AgentEvent(
                                type="thought_action",
                                content=msg.content or "",
                                tool_name=tc.get("name"),
                                tool_args=tc.get("args"),
                            )

                    # Observation step: The tool execution returned results
                    elif isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, "name", "tool")
                        yield AgentEvent(
                            type="observation",
                            content=str(msg.content),
                            tool_name=tool_name,
                        )

                    # Final Answer step: The LLM produced its final response
                    elif isinstance(msg, AIMessage) and not msg.tool_calls:
                        # In voice mode or when thinking is disabled, sanitize any <think> tags
                        answer_text = strip_thinking(msg.content)
                        yield AgentEvent(
                            type="answer",
                            content=answer_text,
                        )

    except Exception:
        # Rollback the last user message if the turn fails so the thread state isn't broken
        if thread_history and thread_history[-1] is user_message:
            thread_history.pop()
        raise
