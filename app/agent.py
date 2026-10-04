"""Agent construction and ReAct graph definition (Phase 1).

Why this file exists:
This module sets up the LLM and compiles the LangGraph ReAct agent.
ReAct (Reason + Act) is an agentic design pattern where the model iteratively:
  1. Reasons about the current conversation and decides if it needs tools.
  2. Acts by calling specific tools with structured arguments.
  3. Observes the tool outputs (observations) returned by the environment.
  4. Repeats until it can produce a final answer for the user.

Framework Note:
In LangGraph 1.2+ / LangChain 1.4+, `langgraph.prebuilt.create_react_agent` is deprecated
in favor of `langchain.agents.create_agent`. Under the hood, `create_agent` compiles into
a LangGraph `CompiledStateGraph` that manages the message state and cyclic transitions.
"""

import os
import re
from collections.abc import Sequence
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from app.tools import (
    calculator,
    get_current_time,
    query_ipl_stats,
    read_mail,
    search_docs,
    search_product,
    send_mail,
    web_search,
)

# Load environment variables from .env file (if present)
load_dotenv()

# ---------------------------------------------------------------------------
# System Prompts for Text vs. Voice Modes
# ---------------------------------------------------------------------------

TEXT_SYSTEM_PROMPT = (
    "You are a helpful AI cricket assistant with access to IPL data (2008-2017).\n"
    "Tool routing rules:\n"
    "- 'search_docs': Match narratives, descriptions, summaries, or specific match events.\n"
    "- 'query_ipl_stats': Numbers, counts, records, aggregates, rankings, or comparisons.\n"
    "- 'web_search': Live news, real-time facts, weather, or topics outside the IPL dataset.\n"
    "- 'get_current_time': Current date, time, or day of week.\n"
    "- 'calculator': Arithmetic calculations.\n"
    "- 'read_mail': Use when the user asks to check, read, or summarise their inbox or recent emails.\n"
    "- 'send_mail': Use when the user asks to send or compose an email; always show the draft preview and wait for explicit user confirmation before calling again with confirm=True.\n"
    "- 'search_product': Use when the user wants to find, compare, or check prices of products on Amazon or Flipkart.\n"
    "Synthesize observations directly. Never hallucinate statistics."
)

VOICE_SYSTEM_PROMPT = (
    "You are an IPL cricket voice assistant. Speak concisely in 1-2 plain sentences.\n"
    "No markdown, bullets, emojis, or code blocks.\n"
    "Tool rules: 'search_docs' for narratives, 'query_ipl_stats' for stats, 'web_search' for live info, 'get_current_time' for time, 'calculator' for math, 'read_mail' to check inbox, 'send_mail' to compose (always confirm before sending)."
)


def strip_thinking(text: str) -> str:
    """Strips <think>...</think> reasoning blocks from model text output.

    Used when ENABLE_THINKING=false or in voice mode as a safeguard if the local
    model outputs reasoning tags.
    """
    if not text:
        return ""
    # Strip complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip any unclosed <think> tag at the end of text
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    # Only strip surrounding whitespace if think-blocks were actually removed,
    # otherwise preserve token-level spaces needed for proper word spacing.
    if cleaned != text:
        return cleaned.strip()
    return cleaned


def get_ollama_llm(
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    enable_thinking: bool = False,
    keep_alive: int | str = -1,
) -> ChatOllama:
    """Instantiate a ChatOllama model with configuration read from environment.

    Args:
        model_name: Model identifier (defaults to OLLAMA_MODEL env var or 'qwen3:8b').
        base_url: Ollama endpoint (defaults to OLLAMA_BASE_URL env var or 'http://localhost:11434').
        temperature: Sampling temperature (default 0 for deterministic tool calling).
        num_ctx: Context window size in tokens (default 4096).
        enable_thinking: If False, sets reasoning=False in ChatOllama to disable thinking tokens.
        keep_alive: Model keep-alive timeout (default -1 to keep loaded indefinitely).

    Returns:
        ChatOllama: Initialized chat model instance.
    """
    model = model_name or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    endpoint = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # langchain-ollama accepts reasoning: bool | str | None which maps to Ollama's 'think' parameter.
    # Setting reasoning=False disables the reasoning/thinking mode for faster responses.
    return ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=temperature,
        num_ctx=num_ctx,
        reasoning=enable_thinking,
        keep_alive=keep_alive,
    )


def build_agent(
    mode: str = "text",
    model_name: str | None = None,
    base_url: str | None = None,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph ReAct agent graph.

    Args:
        mode: Operation mode: 'text' (default) or 'voice' (conversational, short, no markdown).
        model_name: Name of the Ollama model (default 'qwen3:8b').
        base_url: Ollama API URL.
        tools: List of tools to equip the agent with (defaults to demo tools + web search).
        system_prompt: Custom system prompt (overrides mode-based prompt if provided).
        enable_thinking: Explicit override for thinking mode. If None, voice mode defaults
            to False for speed; text mode reads ENABLE_THINKING from .env (default False).

    Returns:
        CompiledStateGraph: The runnable graph ready for .invoke() or .stream().
    """
    # 1. Determine thinking mode
    # Voice mode always has thinking disabled for speed.
    if mode == "voice":
        thinking_enabled = False
    elif enable_thinking is not None:
        thinking_enabled = enable_thinking
    else:
        env_val = os.getenv("ENABLE_THINKING", "false").lower()
        thinking_enabled = env_val in ("true", "1", "yes")

    # 2. Select system prompt
    if system_prompt is not None:
        active_prompt = system_prompt
    elif mode == "voice":
        active_prompt = VOICE_SYSTEM_PROMPT
    else:
        active_prompt = TEXT_SYSTEM_PROMPT

    # 3. Initialize tools (defaults: get_current_time, calculator, web_search, search_docs, query_ipl_stats)
    active_tools = (
        list(tools)
        if tools is not None
        else [get_current_time, calculator, web_search, search_docs, query_ipl_stats, read_mail, send_mail, search_product]
    )

    # 4. Initialize the local Ollama LLM with thinking setting
    llm = get_ollama_llm(
        model_name=model_name,
        base_url=base_url,
        enable_thinking=thinking_enabled,
    )

    # 5. Create the agent using langchain.agents.create_agent (compiles to LangGraph CompiledStateGraph)
    agent = create_agent(
        model=llm,
        tools=active_tools,
        system_prompt=active_prompt,
    )

    return agent
