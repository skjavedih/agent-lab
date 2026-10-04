"""Semantic query router using TypeSafe System One (Jev)."""

import os
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

ROUTE_CRITERIA = {
    "chat": "General chit-chat, conversational greetings, personality responses, or casual questions that require no cricket data or external tools.",
    "ipl_narrative": "Descriptive or storytelling questions about specific IPL cricket matches, finals, historic moments, venue details, or players of the match.",
    "ipl_stats": "Statistical, numeric, aggregate, count, ranking, or record-based IPL cricket questions (e.g. most runs, highest scores, wickets, averages).",
    "web_search": "Questions about current events, live news, real-time facts, weather, or topics outside the historical IPL dataset.",
    "browser_task": "Complex web browsing, navigating websites, multi-step browser automation, clicking links, or filling online forms.",
}


def route(user_text: str) -> str:
    """Classify user query into an execution route using TypeSafe System One (Jev).

    Args:
        user_text: The user's input text to evaluate.

    Returns:
        The selected route option: 'chat', 'ipl_narrative', 'ipl_stats',
        'web_search', or 'browser_task'.
    """
    api_key = os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
    with TypeSafeClient(api_key=api_key) as client:
        response = client.system_one(
            state=user_text,
            questions={
                "route": Choice(
                    instructions="Classify this user request into the single most appropriate handler category.",
                    criteria=ROUTE_CRITERIA,
                )
            },
        )
        return response.choices["route"].choice
