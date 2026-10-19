"""Interactive CLI Chat Loop (Phase 1).

Why this file exists:
Thin command-line wrapper around `app.service.run_turn`.
All orchestration logic and conversation state live in `app/service.py`, allowing
this CLI, a future voice loop (`app/voice.py`), or an API to share the exact same
agent core without code duplication.
"""

import argparse
import os
import sys
import time
from dotenv import load_dotenv
import httpx

from app.service import run_turn

# Load environment configuration
load_dotenv()


def check_ollama_service(base_url: str, model_name: str) -> tuple[bool, str]:
    """Verify that Ollama is reachable and report status."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            available_models = [m.get("name") for m in data.get("models", [])]
            model_base = model_name.split(":")[0]
            found = any(model_name in m or model_base in m for m in available_models)
            if not found:
                return True, (
                    f"⚠️  Ollama is running, but model '{model_name}' was not detected.\n"
                    f"   Available models: {', '.join(available_models) if available_models else 'None'}\n"
                    f"   Run 'ollama pull {model_name}' to download it."
                )
            return True, f"Connected to Ollama at {base_url} (model: {model_name})"
        return False, f"Ollama responded with HTTP status {response.status_code}."
    except Exception as exc:
        return False, (
            f"❌ Unable to connect to Ollama at {base_url}.\n"
            f"   Details: {exc}\n\n"
            f"   Quick Fix:\n"
            f"   1. Start Ollama: `ollama serve` (or open the Ollama desktop app).\n"
            f"   2. Pull the model: `ollama pull {model_name}`\n"
            f"   3. Verify your .env: OLLAMA_BASE_URL={base_url}"
        )


def main():
    """Run the interactive CLI chat loop."""
    parser = argparse.ArgumentParser(
        description="Agent Lab - ReAct Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default="text",
        help="Interaction mode: 'text' (detailed, markdown) or 'voice' (short, plain speech)",
    )
    parser.add_argument(
        "--thread",
        default="default",
        help="Conversation thread ID (used for state isolation and session persistence)",
    )
    args = parser.parse_args()

    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    print("=" * 60)
    print("🤖 Agent Lab - ReAct Agent CLI")
    print(f"Mode: {args.mode.upper()} | Thread ID: '{args.thread}'")
    print("=" * 60)

    # Pre-flight check: ensure Ollama is accessible
    is_connected, status_message = check_ollama_service(base_url, model_name)
    if not is_connected:
        print(f"\n{status_message}\n")
        print("Please start Ollama and try again. Exiting.")
        sys.exit(1)
    else:
        print(f"Status: {status_message}")

    print("\nAgent is ready! Type your message, or 'quit' / 'exit' to stop.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...", flush=True)
        start_time = time.perf_counter()

        try:
            # Delegate execution entirely to the decoupled service layer
            events = run_turn(
                user_text=user_input,
                thread_id=args.thread,
                mode=args.mode,
            )

            answer_started = False
            for event in events:
                if event.type == "thought_action":
                    if answer_started:
                        print()
                        answer_started = False
                    if event.content:
                        print(f"\n[THOUGHT/ACTION] Thought: {event.content}")
                    print(
                        f"[THOUGHT/ACTION] Call tool: {event.tool_name}({event.tool_args})"
                    )

                elif event.type == "observation":
                    if answer_started:
                        print()
                        answer_started = False
                    print(
                        f"[OBSERVATION] Tool '{event.tool_name}' returned: {event.content}"
                    )

                elif event.type == "answer":
                    # Tokens arrive one at a time; print the label once and
                    # append each token on the same line.
                    if not answer_started:
                        print("\n[ANSWER] ", end="", flush=True)
                        answer_started = True
                    print(event.content, end="", flush=True)

            if answer_started:
                print()

            elapsed = time.perf_counter() - start_time
            print(f"\n[⏱️ Response time: {elapsed:.2f}s]")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            print(f"\n❌ Error during execution: {exc}")
            print(f"[⏱️ Elapsed: {elapsed:.2f}s]")
            print("Tip: Check if Ollama is still running and responsive.")


if __name__ == "__main__":
    main()
