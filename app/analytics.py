"""Structured IPL dataset analytics via LLM-assisted pandas queries (Phase 3).

Why this file exists:
Provides the structured analytical path for the IPL assistant. For quantitative,
aggregation, or ranking questions ('how many', 'highest', 'most runs', 'win percentage'),
this module prompts an LLM to generate a minimal pandas snippet against the matches and
deliveries DataFrames and executes it within a restricted sandboxed namespace.
"""

from pathlib import Path
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import pandas as pd

from app.agent import get_ollama_llm, strip_thinking

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATCHES_PATH = DATA_DIR / "matches.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"

# Load both CSVs into module-level DataFrames once (parse date properly)
if MATCHES_PATH.exists():
    matches: pd.DataFrame = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
else:
    matches = pd.DataFrame()

if DELIVERIES_PATH.exists():
    deliveries: pd.DataFrame = pd.read_csv(DELIVERIES_PATH)
else:
    deliveries = pd.DataFrame()


PANDAS_TRANSLATION_PROMPT = """You are an expert Python data analyst. Given a user question about the IPL dataset, write a short, valid Python pandas snippet to answer it.

You have access to two pre-loaded pandas DataFrames:
1. `matches` (636 rows, one row per match from 2008 to 2017):
   Columns: ['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner', 'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs', 'win_by_wickets', 'player_of_match', 'venue', 'umpire1', 'umpire2', 'umpire3']
   - 'season' is an integer year (e.g. 2016).
   - 'date' is a parsed datetime column.
   - 'result' is 'normal', 'tie', or 'no result'.

2. `deliveries` (150,460 rows, one row per ball bowled):
   Columns: ['match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball', 'batsman', 'non_striker', 'bowler', 'is_super_over', 'wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs', 'total_runs', 'player_dismissed', 'dismissal_kind', 'fielder']

Join key: deliveries['match_id'] == matches['id']

Strict Rules:
1. You MUST assign the final answer to a variable named `result`.
2. `result` can be a scalar number, string, Series, or DataFrame.
3. Do NOT import any modules. Do NOT call open, exec, eval, or __import__.
4. Output ONLY the executable Python code snippet. No explanations, no markdown formatting.

Worked Examples:

Example 1:
Question: How many matches has Mumbai Indians won?
Code:
result = int((matches['winner'] == 'Mumbai Indians').sum())

Example 2:
Question: Who scored the most runs overall in IPL history?
Code:
top = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(1)
result = f"{top.index[0]} with {top.iloc[0]} runs"

Example 3:
Question: Which player won the most Player of the Match awards?
Code:
top = matches['player_of_match'].value_counts().head(1)
result = f"{top.index[0]} with {top.iloc[0]} awards"

Example 4:
Question: Which batsman hit the most sixes?
Code:
sixes = deliveries[deliveries['batsman_runs'] == 6]
top = sixes['batsman'].value_counts().head(1)
result = f"{top.index[0]} with {top.iloc[0]} sixes"

Example 5:
Question: Which bowler took the most wickets in IPL?
Code:
w_kinds = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
bowler_w = deliveries[deliveries['dismissal_kind'].isin(w_kinds)]
top = bowler_w['bowler'].value_counts().head(1)
result = f"{top.index[0]} with {top.iloc[0]} wickets"

User Question: {question}
Code:
"""


def _extract_code(text: str) -> str:
    """Clean LLM output and extract raw Python code."""
    cleaned = strip_thinking(text).strip()
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def query_ipl_stats(question: str) -> str:
    """Generate and execute a sandboxed pandas query against matches and deliveries.

    Security & Sandboxing Note:
    This function executes LLM-generated Python code and is sandboxed for that reason:
      - The execution namespace is restricted strictly to `pd`, `matches`, and `deliveries`.
      - Builtin functions like open, exec, eval, __import__, compile are completely omitted
        to prevent file-system access, network calls, or arbitrary code execution.
      - We will harden this further with OS/process isolation in Phase 7.
    """
    if matches.empty or deliveries.empty:
        return "Error: IPL dataset DataFrames are not loaded. Please verify data/ directory."

    prompt = PANDAS_TRANSLATION_PROMPT.replace("{question}", question)

    try:
        llm = get_ollama_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        code = _extract_code(response.content)
    except Exception as exc:
        return f"Error communicating with Ollama to generate pandas query: {exc}"

    # Sandbox: inspect for forbidden statements/tokens
    forbidden_tokens = [
        "__import__",
        "import ",
        "open(",
        "exec(",
        "eval(",
        "compile(",
        "os.",
        "sys.",
        "subprocess",
        "shutil",
        "socket",
    ]
    for token in forbidden_tokens:
        if token in code:
            return (
                f"Security Error: Generated code contains forbidden token '{token}'."
            )

    # Safe builtins: allow standard Python primitives while omitting dangerous I/O and dynamic execution
    safe_builtins = {
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "None": None,
        "True": True,
        "False": False,
    }

    local_namespace = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "matches": matches,
        "deliveries": deliveries,
    }

    try:
        exec(code, local_namespace)
        if "result" not in local_namespace:
            return (
                f"Query completed without assigning a 'result' variable.\n"
                f"Generated code was:\n{code}"
            )

        result_val = local_namespace["result"]
        if isinstance(result_val, (pd.DataFrame, pd.Series)):
            return result_val.to_string()
        return str(result_val)

    except Exception as exc:
        # Return error message as output instead of crashing so agent can see it failed
        return (
            f"Error executing pandas query: {type(exc).__name__}: {exc}\n"
            f"Generated code was:\n{code}"
        )
