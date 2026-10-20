"""LangGraph HITL demo: one ReAct agent, tool approval, durable permissions.

Run from the repo root:

    python workshop/langgraph_hitl_permissions_demo.py

Lightweight checks (no Ollama):

    python workshop/langgraph_hitl_permissions_demo.py --self-test

Graph pauses with interrupt() and resumes with Command(resume=...).
Checkpoints live in SQLite so a paused approval survives a restart.
Allow/deny rules live in a separate SQLite file and survive new sessions.

langgraph-checkpoint-sqlite is not installed. This file includes a small
stdlib SQLite checkpointer that follows the installed saver contract.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Iterator, TypedDict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

load_dotenv()

# Same Ollama settings as the rest of the project. The model is read when
# the agent node runs (see _llm_agent), not hardcoded here.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")

_DATA_DIR = Path(os.getenv("HITL_DATA_DIR", str(Path(_ROOT) / "workshop" / "hitl_data")))
PERMISSIONS_DB = Path(
    os.getenv("HITL_PERMISSIONS_DB", str(_DATA_DIR / "permissions.sqlite"))
)
CHECKPOINT_DB = Path(
    os.getenv("HITL_CHECKPOINT_DB", str(_DATA_DIR / "checkpoints.sqlite"))
)
DEMO_WORKSPACE = Path(
    os.getenv("HITL_DEMO_WORKSPACE", str(Path(_ROOT) / "workshop" / "hitl_demo_workspace"))
)

MAX_TOOL_OUTPUT = 700
SHELL_BINS = {"ls": "/bin/ls", "pwd": "/bin/pwd"}
SECRET_NAMES = {".env", ".env.local", ".env.example", "credentials.json", "secrets.json"}

EXAMPLES = [
    "Search the cricket RAG data for IPL information.",
    "Search the web for the latest LangGraph documentation.",
    "List the files in the configured demo workspace.",
    "Create a small text file in the demo workspace.",
    "Show the approval process for a restricted Shell command.",
]

SYSTEM_PROMPT = (
    "Workshop assistant. If a tool is needed, call it immediately and write no answer text yet. "
    "After the tool result, answer in at most three short sentences. "
    "cricket_rag_search: IPL history. tavily_web_search: live web. "
    "shell: only pwd or ls in the demo workspace. file_read and file_write: only that workspace. "
    f"Workspace: {DEMO_WORKSPACE}. "
    "List files -> shell command ls. "
    "Create a small text file -> file_write path workshop-note.txt content 'hello from the HITL demo'. "
    "Restricted shell approval demo -> shell command 'rm -rf /tmp/hitl-demo' only. "
    "If status is rejected, say it was blocked. Do not reuse an older tool result."
)

TOOL_PURPOSE = {
    "cricket_rag_search": "Search the existing cricket match RAG index.",
    "tavily_web_search": "Search the live web with the existing Tavily tool.",
    "shell": "Run one allowlisted read-only command in the demo workspace.",
    "file_read": "Read one file inside the demo workspace.",
    "file_write": "Write one file inside the demo workspace.",
}

SCOPE_TEXT = {
    ("cricket_rag_search", "search"): "Allow every cricket_rag_search call (category: search).",
    ("tavily_web_search", "search"): "Allow every tavily_web_search call (category: search).",
    ("shell", "read_only"): (
        "Allow read-only shell in the demo workspace: pwd and ls only. "
        "This does not allow other commands."
    ),
    ("file_read", "read"): "Allow file_read for files inside the demo workspace (secrets still blocked).",
    ("file_write", "write"): "Allow file_write inside the demo workspace (secrets still blocked).",
    ("shell", "blocked"): (
        "This command is outside the read-only allowlist (pwd, ls). "
        "Approve Once and Approve All will not run it."
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ver_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _blob(serde: Any, value: Any) -> tuple[str, bytes]:
    typ, raw = serde.dumps_typed(value)
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    return typ, bytes(raw)


def integration_banner() -> str:
    """Short status for the workshop UI. Does not invent missing integrations."""
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_key_here":
        tavily = (
            "Tavily: no API key — `tavily_web_search` calls the existing tool, "
            "which returns a disabled message. No substitute search is used."
        )
    else:
        tavily = "Tavily: API key is set. `tavily_web_search` uses `app.tools.web_search`."
    return (
        f"**Ollama:** `{OLLAMA_MODEL}` @ `{OLLAMA_BASE_URL}`  \n"
        f"**Cricket RAG:** `app.tools.search_docs` (vector backend `{VECTOR_BACKEND}`)  \n"
        f"**{tavily}**  \n"
        f"**Workspace:** `{DEMO_WORKSPACE}`  \n"
        f"**Permission SQLite:** `{PERMISSIONS_DB}`  \n"
        f"**Checkpoint SQLite:** `{CHECKPOINT_DB}`  \n"
        "Checkpointer package `langgraph-checkpoint-sqlite` is not installed; "
        "this demo uses a local SQLite saver in this file. "
        "Permission rules are a separate database from graph checkpoints."
    )


# ---------------------------------------------------------------------------
# Permission rules (separate from graph checkpoints)
# ---------------------------------------------------------------------------


class PermissionStore:
    """Durable allow/deny rules. Deny wins over allow for the same scope."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS permission_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                action_category TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('allow', 'deny')),
                persistence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(tool_name, action_category, decision)
            )
            """
        )
        # Side-effect ledger so a replayed interrupt node does not run a tool twice.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS executed_requests (
                request_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def lookup(self, tool_name: str, category: str) -> str | None:
        """Return 'deny', 'allow', or None. A deny row beats an allow row."""
        with self._lock:
            denied = self._conn.execute(
                """
                SELECT 1 FROM permission_rules
                WHERE tool_name=? AND action_category=? AND decision='deny'
                """,
                (tool_name, category),
            ).fetchone()
            if denied:
                return "deny"
            allowed = self._conn.execute(
                """
                SELECT 1 FROM permission_rules
                WHERE tool_name=? AND action_category=? AND decision='allow'
                """,
                (tool_name, category),
            ).fetchone()
            return "allow" if allowed else None

    def save_rule(self, tool_name: str, category: str, decision: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO permission_rules
                    (tool_name, action_category, decision, persistence, created_at)
                VALUES (?, ?, ?, 'permanent', ?)
                ON CONFLICT(tool_name, action_category, decision) DO NOTHING
                """,
                (tool_name, category, decision, _now()),
            )
            self._conn.commit()

    def list_rules(self) -> list[tuple]:
        with self._lock:
            return self._conn.execute(
                """
                SELECT tool_name, action_category, decision, persistence, created_at
                FROM permission_rules
                ORDER BY created_at, id
                """
            ).fetchall()

    def clear_rules(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM permission_rules")
            # Drop cached tool results too. Otherwise a later call can show an
            # old result before the next approval pause.
            self._conn.execute("DELETE FROM executed_requests")
            self._conn.commit()
            return cur.rowcount

    def get_done(self, request_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, result_json FROM executed_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        if row and row[0] == "done":
            return row[1]
        return None

    def try_begin(self, request_id: str, tool_name: str) -> bool:
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO executed_requests
                        (request_id, tool_name, status, result_json, created_at)
                    VALUES (?, ?, 'running', '', ?)
                    """,
                    (request_id, tool_name, _now()),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def finish(self, request_id: str, result_json: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE executed_requests
                SET status='done', result_json=?
                WHERE request_id=?
                """,
                (result_json, request_id),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def format_rules(store: PermissionStore) -> str:
    rows = store.list_rules()
    if not rows:
        return "No saved allow/deny rules."
    lines = ["tool | category | decision | persistence | created"]
    for tool_name, category, decision, persistence, created in rows:
        lines.append(f"{tool_name} | {category} | {decision} | {persistence} | {created}")
    lines.append("Precedence: deny beats allow for the same tool and category.")
    return "\n".join(lines)


def clear_permission_rules(store: PermissionStore, confirmed: bool) -> str:
    if not confirmed:
        return "Confirmation required. Saved rules were not changed."
    removed = store.clear_rules()
    return f"Removed {removed} saved allow/deny rule(s). Checkpoints were kept."


# ---------------------------------------------------------------------------
# Durable graph checkpointer (stdlib sqlite3)
# ---------------------------------------------------------------------------


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """Small durable saver. Mirrors the in-memory saver's put/get/writes shape.

    HITL note: interrupt() persists its pause in this store. Resume loads that
    checkpoint and continues the same node. This is not the permission database.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                checkpoint_blob BLOB NOT NULL,
                metadata_type TEXT NOT NULL,
                metadata_blob BLOB NOT NULL,
                parent_id TEXT,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                write_idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                task_path TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        parent_id = configurable.get("checkpoint_id")
        copied = checkpoint.copy()
        values: dict[str, Any] = copied.pop("channel_values")  # type: ignore[misc]
        checkpoint_type, checkpoint_blob = _blob(self.serde, copied)
        meta_type, meta_blob = _blob(
            self.serde, get_checkpoint_metadata(config, metadata)
        )
        with self._lock:
            for channel, version in new_versions.items():
                if channel in values:
                    value_type, value_blob = _blob(self.serde, values[channel])
                else:
                    value_type, value_blob = "empty", b""
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO channel_blobs
                        (thread_id, checkpoint_ns, channel, version, value_type, value_blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        channel,
                        _ver_key(version),
                        value_type,
                        value_blob,
                    ),
                )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, checkpoint_type,
                     checkpoint_blob, metadata_type, metadata_blob, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_type,
                    checkpoint_blob,
                    meta_type,
                    meta_blob,
                    parent_id,
                ),
            )
            self._conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]
        with self._lock:
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                if write_idx >= 0:
                    existing = self._conn.execute(
                        """
                        SELECT 1 FROM writes
                        WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                          AND task_id=? AND write_idx=?
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx),
                    ).fetchone()
                    if existing:
                        continue
                else:
                    self._conn.execute(
                        """
                        DELETE FROM writes
                        WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                          AND task_id=? AND write_idx=?
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx),
                    )
                value_type, value_blob = _blob(self.serde, value)
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx,
                         channel, value_type, value_blob, task_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_idx,
                        channel,
                        value_type,
                        value_blob,
                        task_path,
                    ),
                )
            self._conn.commit()

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: dict[str, Any]
    ) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for channel, version in versions.items():
            row = self._conn.execute(
                """
                SELECT value_type, value_blob FROM channel_blobs
                WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?
                """,
                (thread_id, checkpoint_ns, channel, _ver_key(version)),
            ).fetchone()
            if not row or row[0] == "empty":
                continue
            loaded[channel] = self.serde.loads_typed((row[0], row[1]))
        return loaded

    def _tuple_from_row(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        checkpoint_type: str,
        checkpoint_blob: bytes,
        metadata_type: str,
        metadata_blob: bytes,
        parent_id: str | None,
    ) -> CheckpointTuple:
        stored = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
        writes = self._conn.execute(
            """
            SELECT task_id, channel, value_type, value_blob
            FROM writes
            WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
            ORDER BY rowid
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        pending = [
            (task_id, channel, self.serde.loads_typed((value_type, value_blob)))
            for task_id, channel, value_type, value_blob in writes
        ]
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint={
                **stored,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, stored.get("channel_versions", {})
                ),
            },
            metadata=self.serde.loads_typed((metadata_type, metadata_blob)),
            pending_writes=pending,
            parent_config=parent_config,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self._lock:
            if checkpoint_id:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint_type, checkpoint_blob,
                           metadata_type, metadata_blob, parent_id
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint_type, checkpoint_blob,
                           metadata_type, metadata_blob, parent_id
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
            if not row:
                return None
            return self._tuple_from_row(thread_id, checkpoint_ns, *row)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_type, "
            "checkpoint_blob, metadata_type, metadata_blob, parent_id FROM checkpoints"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id:
            clauses.append("thread_id=?")
            params.append(thread_id)
            clauses.append("checkpoint_ns=?")
            params.append(checkpoint_ns)
        if before_id:
            clauses.append("checkpoint_id < ?")
            params.append(before_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
            tuples = [self._tuple_from_row(*row) for row in rows]
        for item in tuples:
            if filter and any(item.metadata.get(key) != val for key, val in filter.items()):
                continue
            yield item

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            for table in ("checkpoints", "channel_blobs", "writes"):
                self._conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
            self._conn.commit()


# ---------------------------------------------------------------------------
# Tools: validate, then authorize, then execute
# ---------------------------------------------------------------------------


@dataclass
class ValidatedCall:
    ok: bool
    tool: str
    category: str
    reason: str
    args: dict[str, Any]
    normalized: dict[str, Any]


def ensure_workspace(path: Path | None = None) -> Path:
    workspace = path or DEMO_WORKSPACE
    workspace.mkdir(parents=True, exist_ok=True)
    welcome = workspace / "welcome.txt"
    if not welcome.exists():
        welcome.write_text("HITL demo workspace. Safe to list and read.\n", encoding="utf-8")
    return workspace


def _is_secret(path: Path) -> bool:
    name = path.name
    if name in SECRET_NAMES or name.startswith(".env"):
        return True
    return name.endswith((".pem", ".key"))


def _resolve_workspace_file(raw: str, *, workspace: Path) -> tuple[Path | None, str]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "Path is empty."
    if "\x00" in raw or len(raw) > 200:
        return None, "Path is not valid."
    if raw.startswith("~"):
        return None, "Home-directory paths are not allowed."
    parts = Path(raw).parts
    if ".." in parts:
        return None, "Path traversal is not allowed."
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        return None, "Path is outside the demo workspace."
    if _is_secret(resolved) or any(_is_secret(parent) for parent in resolved.parents):
        return None, "Refusing to touch env files or other secrets."
    return resolved, ""


def validate_call(tool_name: str, args: Any, *, workspace: Path) -> ValidatedCall:
    """Reject malformed or out-of-scope calls before any approval or execution."""
    if tool_name not in TOOL_PURPOSE:
        return ValidatedCall(False, tool_name, "", f"Unknown tool '{tool_name}'.", {}, {})
    if not isinstance(args, dict):
        return ValidatedCall(False, tool_name, "", "Tool arguments must be an object.", {}, {})

    if tool_name in {"cricket_rag_search", "tavily_web_search"}:
        if set(args) != {"query"} or not isinstance(args.get("query"), str):
            return ValidatedCall(False, tool_name, "search", "Expected {query: string}.", args, {})
        query = args["query"].strip()
        if not query or len(query) > 500:
            return ValidatedCall(False, tool_name, "search", "Query is empty or too long.", args, {})
        return ValidatedCall(True, tool_name, "search", "", args, {"query": query})

    if tool_name == "shell":
        return _validate_shell(args, workspace=workspace)

    if tool_name == "file_read":
        if set(args) != {"path"}:
            return ValidatedCall(False, tool_name, "read", "Expected {path: string}.", args, {})
        resolved, reason = _resolve_workspace_file(args.get("path", ""), workspace=workspace)
        if resolved is None:
            return ValidatedCall(False, tool_name, "read", reason, args, {})
        return ValidatedCall(True, tool_name, "read", "", args, {"path": str(resolved)})

    if set(args) != {"path", "content"}:
        return ValidatedCall(
            False, tool_name, "write", "Expected {path: string, content: string}.", args, {}
        )
    if not isinstance(args.get("content"), str) or len(args["content"]) > 8000:
        return ValidatedCall(False, tool_name, "write", "Content must be a short string.", args, {})
    resolved, reason = _resolve_workspace_file(args.get("path", ""), workspace=workspace)
    if resolved is None:
        return ValidatedCall(False, tool_name, "write", reason, args, {})
    return ValidatedCall(True, tool_name, "write", "", args, {"path": str(resolved), "content": args["content"]})


def _validate_shell(args: dict, *, workspace: Path) -> ValidatedCall:
    if set(args) != {"command"} or not isinstance(args.get("command"), str):
        return ValidatedCall(False, "shell", "read_only", "Expected {command: string}.", args, {})
    command = args["command"].strip()
    if not command or any(ch in command for ch in "|&;`$<>\n"):
        return ValidatedCall(
            False,
            "shell",
            "read_only",
            "Command is empty or contains shell metacharacters. Not executed.",
            args,
            {},
        )
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return ValidatedCall(False, "shell", "read_only", f"Could not parse command: {exc}", args, {})
    if not parts or parts[0] not in SHELL_BINS:
        name = parts[0] if parts else ""
        # Parsed, but not allowlisted. The gate can show this for approval and
        # still refuses to execute it. Metacharacters never reach this branch.
        return ValidatedCall(
            False,
            "shell",
            "blocked",
            f"'{name}' is outside the read-only allowlist (pwd, ls). Not executed.",
            args,
            {},
        )
    binary = SHELL_BINS[parts[0]]
    if not Path(binary).exists():
        return ValidatedCall(False, "shell", "read_only", f"{binary} is not on this machine.", args, {})
    if parts[0] == "pwd":
        if len(parts) != 1:
            return ValidatedCall(False, "shell", "read_only", "pwd takes no arguments.", args, {})
        return ValidatedCall(True, "shell", "read_only", "", args, {"argv": [binary]})
    flags: list[str] = []
    paths: list[str] = []
    for part in parts[1:]:
        if part.startswith("-"):
            if part not in {"-l", "-a", "-la", "-al"}:
                return ValidatedCall(False, "shell", "read_only", f"Flag {part} is not allowed.", args, {})
            flags.append(part)
        else:
            paths.append(part)
    if len(paths) > 1:
        return ValidatedCall(False, "shell", "read_only", "ls accepts at most one path.", args, {})
    target = paths[0] if paths else "."
    resolved, reason = _resolve_workspace_file(target, workspace=workspace)
    if resolved is None:
        return ValidatedCall(False, "shell", "read_only", reason, args, {})
    if not resolved.exists() or not resolved.is_dir():
        return ValidatedCall(False, "shell", "read_only", "ls path must be a directory in the workspace.", args, {})
    return ValidatedCall(True, "shell", "read_only", "", args, {"argv": [binary, *flags, str(resolved)]})


EXEC_LOG: list[str] = []


def _result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + "\n...[truncated]"


def execute_authorized(validated: ValidatedCall, *, workspace: Path) -> str:
    """Run a call that already passed validation and authorization."""
    tool_name = validated.tool
    EXEC_LOG.append(tool_name)
    try:
        if tool_name == "cricket_rag_search":
            from app.tools import search_docs

            output = search_docs.invoke({"query": validated.normalized["query"]})
        elif tool_name == "tavily_web_search":
            from app.tools import web_search

            output = web_search.invoke({"query": validated.normalized["query"]})
            if str(output).startswith("Error: TAVILY_API_KEY"):
                output = (
                    "Tavily unavailable. The existing web_search tool did not run a live search. "
                    + str(output)
                )
        elif tool_name == "shell":
            proc = subprocess.run(
                validated.normalized["argv"],
                cwd=str(workspace),
                shell=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": "/bin:/usr/bin", "LC_ALL": "C"},
            )
            output = f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}"
        elif tool_name == "file_read":
            path = Path(validated.normalized["path"])
            if not path.is_file():
                return _result(
                    {
                        "status": "rejected",
                        "tool": tool_name,
                        "decision": "validation",
                        "executed": False,
                        "reason": "Path is not a file in the demo workspace.",
                    }
                )
            output = path.read_text(encoding="utf-8", errors="replace")
        else:
            path = Path(validated.normalized["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.parent.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Write path escaped the demo workspace.")
            path.write_text(validated.normalized["content"], encoding="utf-8")
            output = f"Wrote {path.name} ({len(validated.normalized['content'])} chars)."
    except Exception as exc:
        return _result(
            {
                "status": "rejected",
                "tool": tool_name,
                "decision": "execution_error",
                "executed": False,
                "reason": str(exc),
            }
        )
    return _result(
        {
            "status": "ok",
            "tool": tool_name,
            "category": validated.category,
            "executed": True,
            "output": _clip(str(output)),
        }
    )


def _execute_once(
    store: PermissionStore,
    request_id: str,
    validated: ValidatedCall,
    *,
    workspace: Path,
    decision: str,
) -> str:
    """Execute at most once per request id, even if the node is replayed."""
    prior = store.get_done(request_id)
    if prior is not None:
        return prior
    if not store.try_begin(request_id, validated.tool):
        prior = store.get_done(request_id)
        if prior is not None:
            return prior
        return _result(
            {
                "status": "rejected",
                "tool": validated.tool,
                "decision": "duplicate",
                "executed": False,
                "reason": "This request already started. It was not executed again.",
            }
        )
    result = execute_authorized(validated, workspace=workspace)
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        payload = {"status": "ok", "output": result}
    payload["decision"] = decision
    result = _result(payload)
    store.finish(request_id, result)
    return result


def _reject(tool_name: str, decision: str, reason: str, args: dict) -> str:
    return _result(
        {
            "status": "rejected",
            "tool": tool_name,
            "decision": decision,
            "executed": False,
            "reason": reason,
            "arguments": args,
        }
    )


def resolve_tool_call(
    tool_call: dict,
    *,
    thread_id: str,
    store: PermissionStore,
    workspace: Path,
) -> str:
    """Permission gate for one proposed call.

    interrupt() pauses before any side effect. On resume the node starts
    again, interrupt() returns the human decision, and only then may the
    tool run. A saved allow rule skips the pause. A saved deny rule blocks.
    """
    tool_name = str(tool_call.get("name") or "")
    args = tool_call.get("args") or {}
    call_id = str(tool_call.get("id") or uuid.uuid4().hex)
    request_id = f"{thread_id}:{call_id}"

    validated = validate_call(tool_name, args, workspace=workspace)
    public_args = args if isinstance(args, dict) else {}
    if not validated.ok and validated.category != "blocked":
        return _reject(tool_name, "validation", validated.reason, public_args)
    if validated.category == "blocked":
        # Show the real command in the approval panel. No decision can run it.
        if store.lookup("shell", "blocked") == "deny":
            return _reject(
                tool_name,
                "deny_rule",
                "Permanent deny for shell / blocked. The command was not executed.",
                public_args,
            )
        decision = interrupt(
            {
                "request_id": request_id,
                "tool": tool_name,
                "args": public_args,
                "purpose": "Restricted shell command. The allowlist blocks execution.",
                "category": "blocked",
                "scope": SCOPE_TEXT[("shell", "blocked")],
            }
        )
        if not isinstance(decision, dict) or decision.get("request_id") != request_id:
            return _reject(tool_name, "stale", "Stale or duplicate approval was ignored.", public_args)
        if decision.get("decision") == "deny_permanent":
            store.save_rule("shell", "blocked", "deny")
        return _reject(
            tool_name,
            "allowlist",
            validated.reason + " Approval cannot override the allowlist.",
            public_args,
        )

    rule = store.lookup(tool_name, validated.category)
    if rule == "deny":
        return _reject(
            tool_name,
            "deny_rule",
            f"Permanent deny for {tool_name} / {validated.category}.",
            args,
        )
    if rule == "allow":
        return _execute_once(
            store, request_id, validated, workspace=workspace, decision="allow_rule"
        )

    # No side effects above this line on the approval path. The first call
    # pauses; the replay receives `decision` and only then executes.
    decision = interrupt(
        {
            "request_id": request_id,
            "tool": tool_name,
            "args": args,
            "purpose": TOOL_PURPOSE[tool_name],
            "category": validated.category,
            "scope": SCOPE_TEXT[(tool_name, validated.category)],
        }
    )
    if not isinstance(decision, dict) or decision.get("request_id") != request_id:
        return _reject(tool_name, "stale", "Stale or duplicate approval was ignored.", args)

    choice = decision.get("decision")
    if choice == "approve_all":
        store.save_rule(tool_name, validated.category, "allow")
    elif choice == "deny_permanent":
        store.save_rule(tool_name, validated.category, "deny")

    if store.lookup(tool_name, validated.category) == "deny" or choice in {
        "reject_once",
        "deny_permanent",
    }:
        reason = "You rejected this action."
        if choice == "deny_permanent" or store.lookup(tool_name, validated.category) == "deny":
            reason = (
                f"Denied {tool_name} / {validated.category}. "
                "A deny rule takes precedence over an allow rule."
            )
        return _reject(tool_name, choice or "rejected", reason, args)

    if choice in {"approve_once", "approve_all"}:
        label = "approve_once" if choice == "approve_once" else "approve_all"
        return _execute_once(
            store, request_id, validated, workspace=workspace, decision=label
        )
    return _reject(tool_name, "invalid_decision", "Unknown approval decision.", args)


# Schema-only tools. The graph never calls these functions; the gate does.
@tool
def cricket_rag_search(query: str) -> str:
    """Search stored IPL cricket match narratives."""
    raise RuntimeError("cricket_rag_search runs only through the permission gate")


@tool
def tavily_web_search(query: str) -> str:
    """Search the live web with Tavily."""
    raise RuntimeError("tavily_web_search runs only through the permission gate")


@tool
def shell(command: str) -> str:
    """Run pwd or ls inside the demo workspace. Other commands are rejected."""
    raise RuntimeError("shell runs only through the permission gate")


@tool
def file_read(path: str) -> str:
    """Read a file inside the demo workspace."""
    raise RuntimeError("file_read runs only through the permission gate")


@tool
def file_write(path: str, content: str) -> str:
    """Write a text file inside the demo workspace."""
    raise RuntimeError("file_write runs only through the permission gate")


AGENT_TOOLS = [cricket_rag_search, tavily_web_search, shell, file_read, file_write]


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content)


def _strip(text: str) -> str:
    from app.agent import strip_thinking

    return strip_thinking(text)


def _current_turn(messages: list) -> list:
    """Messages from the latest user task onward."""
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index
    return messages[start:]


def messages_for_display(messages: list, pending: dict | None) -> list:
    """While approval is pending, hide earlier answers.

    A reused thread still contains the previous tool answer. Showing it above
    the new pause looks like the tool already ran.
    """
    if pending:
        return _current_turn(messages)
    return messages


def _messages_for_model(messages: list) -> list:
    """Send only the system prompt and the current turn.

    Older tool dumps make every later request slower and tempt the model to
    repeat the previous answer before the new approval.
    """
    system = [message for message in messages if isinstance(message, SystemMessage)][:1]
    return system + _current_turn(messages)


def _llm_agent(state: GraphState) -> dict:
    from app.agent import get_ollama_llm

    # Thinking off and a short reply cap. This demo only needs a tool choice
    # or a few sentences, using the same Ollama model as the rest of the app.
    llm = get_ollama_llm(
        temperature=0.0,
        enable_thinking=False,
        num_ctx=2048,
    )
    llm.num_predict = 220
    response = llm.bind_tools(AGENT_TOOLS).invoke(_messages_for_model(state["messages"]))
    if isinstance(response.content, str):
        response.content = _strip(response.content)
    return {"messages": [response]}


def build_graph(
    checkpointer: BaseCheckpointSaver,
    store: PermissionStore,
    *,
    agent_fn=None,
    workspace: Path | None = None,
):
    """ReAct loop: agent proposes a tool, the gate authorizes it, then the agent answers."""
    workspace = workspace or DEMO_WORKSPACE

    def agent_node(state: GraphState) -> dict:
        if agent_fn is not None:
            return agent_fn(state)
        return _llm_agent(state)

    def permission_gate(state: GraphState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        thread_id = config["configurable"]["thread_id"]
        tool_messages = []
        for tool_call in last.tool_calls or []:
            content = resolve_tool_call(
                tool_call,
                thread_id=thread_id,
                store=store,
                workspace=workspace,
            )
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": tool_messages}

    def route(state: GraphState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "permission_gate"
        return END

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("permission_gate", permission_gate)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"permission_gate": "permission_gate", END: END})
    graph.add_edge("permission_gate", "agent")
    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}


def pending_payload(graph, thread_id: str) -> dict | None:
    snapshot = graph.get_state(thread_config(thread_id))
    if not snapshot.interrupts:
        return None
    value = snapshot.interrupts[0].value
    return value if isinstance(value, dict) else {"raw": value}


def apply_decision(graph, thread_id: str, request_id: str, decision: str) -> str:
    """Resume a paused graph. Rejects stale ids and duplicate clicks."""
    pending = pending_payload(graph, thread_id)
    if not pending:
        return "No pending approval. Duplicate submission ignored."
    if pending.get("request_id") != request_id:
        return "Stale approval ignored. The pending request id does not match."
    graph.invoke(
        Command(resume={"request_id": request_id, "decision": decision}),
        thread_config(thread_id),
        durability="sync",
    )
    return f"Recorded {decision} for {request_id}."


def chat_from_messages(messages: list, pending: dict | None) -> list[dict]:
    history: list[dict] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": _text(message.content)})
        elif isinstance(message, AIMessage):
            text = _text(message.content).strip()
            if text:
                history.append({"role": "assistant", "content": text})
    if pending:
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Paused before running `{pending.get('tool')}`. "
                    "Choose Approve Once, Approve All, Reject Once, or Deny Permanently."
                ),
            }
        )
    return history


def activity_from_messages(messages: list, pending: dict | None) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                lines.append(
                    f"tool_call {tool_call['name']} {json.dumps(tool_call.get('args'), ensure_ascii=False)}"
                )
        if isinstance(message, ToolMessage):
            preview = str(message.content).replace("\n", " ")
            lines.append(f"result {message.name}: {preview[:500]}")
    if pending:
        lines.append(
            "PAUSED "
            f"request={pending.get('request_id')} "
            f"tool={pending.get('tool')} "
            f"category={pending.get('category')}"
        )
        lines.append(f"purpose: {pending.get('purpose')}")
        lines.append(f"Approve All scope: {pending.get('scope')}")
        lines.append(f"args: {json.dumps(pending.get('args'), ensure_ascii=False)}")
    return "\n".join(lines) if lines else "(no activity yet)"


def submit_task(graph, store: PermissionStore, thread_id: str, message: str) -> str:
    if pending_payload(graph, thread_id):
        return "An approval is already pending. Decide on it before sending another task."
    snapshot = graph.get_state(thread_config(thread_id))
    existing = (snapshot.values or {}).get("messages") if snapshot else None
    payload: list = [HumanMessage(content=message)]
    if not existing:
        payload.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    graph.invoke({"messages": payload}, thread_config(thread_id), durability="sync")
    return "Task updated."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui():
    ensure_workspace()
    store = PermissionStore(PERMISSIONS_DB)
    saver = SqliteCheckpointSaver(CHECKPOINT_DB)
    graph = build_graph(saver, store)

    def view(thread_id: str, notice: str):
        pending = pending_payload(graph, thread_id)
        messages = graph.get_state(thread_config(thread_id)).values.get("messages") or []
        shown = messages_for_display(messages, pending)
        return (
            chat_from_messages(shown, pending),
            pending.get("tool", "") if pending else "",
            json.dumps(pending.get("args"), indent=2) if pending else "",
            pending.get("purpose", "") if pending else "Nothing is waiting for approval.",
            pending.get("scope", "") if pending else "",
            pending.get("request_id", "") if pending else "",
            format_rules(store),
            activity_from_messages(shown, pending),
            notice,
        )

    def on_send(message: str, thread_id: str):
        text = (message or "").strip()
        if not text:
            return ("", *view(thread_id, "Enter a task first."))
        notice = submit_task(graph, store, thread_id, text)
        return ("", *view(thread_id, notice))

    def on_decision(decision: str, thread_id: str, request_id: str):
        if not request_id:
            return view(thread_id, "No pending approval.")
        notice = apply_decision(graph, thread_id, request_id, decision)
        return view(thread_id, notice)

    def start_fresh_thread(reason: str):
        # A new id every time, including browser refresh. Reusing hitl-session-1
        # reloaded the old checkpoint and painted its answer before the new pause.
        thread_id = "hitl-" + uuid.uuid4().hex[:12]
        return (
            thread_id,
            [],
            "",
            "",
            "Nothing is waiting for approval.",
            "",
            "",
            format_rules(store),
            "(no activity yet)",
            f"{reason} {thread_id}. Saved permission rules were kept.",
        )

    def on_clear(confirmed: bool):
        notice = clear_permission_rules(store, confirmed)
        rules = format_rules(store)
        return rules, notice, False

    with gr.Blocks(title="LangGraph HITL Permissions") as demo:
        gr.Markdown(integration_banner())
        gr.Markdown(
            "Approve Once runs only this request. "
            "Approve All saves an allow rule for the scope shown below. "
            "Reject Once blocks this request. "
            "Deny Permanently saves a deny rule. Deny wins over allow."
        )
        thread_id = gr.Textbox(value="", label="Thread ID (Session ID)")
        example = gr.Dropdown(EXAMPLES, label="Example tasks", value=None)
        chatbot = gr.Chatbot(height=360, label="Conversation")
        msg = gr.Textbox(label="Task", placeholder="Ask for a tool action, or pick an example")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            new_session = gr.Button("New session")
        gr.Markdown("### Pending tool approval")
        tool_name = gr.Textbox(label="Tool", interactive=False)
        tool_args = gr.Textbox(label="Proposed arguments", lines=4, interactive=False)
        purpose = gr.Textbox(label="Purpose", lines=2, interactive=False)
        scope = gr.Textbox(label="Approve All scope", lines=2, interactive=False)
        request_id = gr.Textbox(label="Request ID", interactive=False)
        with gr.Row():
            approve_once = gr.Button("Approve Once")
            approve_all = gr.Button("Approve All")
            reject_once = gr.Button("Reject Once")
            deny_permanent = gr.Button("Deny Permanently")
        gr.Markdown("### Saved permission rules")
        rules = gr.Textbox(value=format_rules(store), label="Allow / deny rules", lines=8)
        confirm_clear = gr.Checkbox(label="I confirm clearing all saved allow/deny rules", value=False)
        clear = gr.Button("Clear saved permissions")
        activity = gr.Textbox(label="Activity log", lines=12)
        notice = gr.Textbox(label="Status", lines=2)

        example.change(lambda value: value or "", example, msg)
        send.click(
            on_send,
            inputs=[msg, thread_id],
            outputs=[msg, chatbot, tool_name, tool_args, purpose, scope, request_id, rules, activity, notice],
        )
        msg.submit(
            on_send,
            inputs=[msg, thread_id],
            outputs=[msg, chatbot, tool_name, tool_args, purpose, scope, request_id, rules, activity, notice],
        )
        decision_outputs = [chatbot, tool_name, tool_args, purpose, scope, request_id, rules, activity, notice]
        approve_once.click(
            lambda thread, req: on_decision("approve_once", thread, req),
            inputs=[thread_id, request_id],
            outputs=decision_outputs,
        )
        approve_all.click(
            lambda thread, req: on_decision("approve_all", thread, req),
            inputs=[thread_id, request_id],
            outputs=decision_outputs,
        )
        reject_once.click(
            lambda thread, req: on_decision("reject_once", thread, req),
            inputs=[thread_id, request_id],
            outputs=decision_outputs,
        )
        deny_permanent.click(
            lambda thread, req: on_decision("deny_permanent", thread, req),
            inputs=[thread_id, request_id],
            outputs=decision_outputs,
        )
        fresh_outputs = [
            thread_id, chatbot, tool_name, tool_args, purpose, scope, request_id, rules, activity, notice,
        ]
        demo.load(lambda: start_fresh_thread("Browser session"), outputs=fresh_outputs)
        new_session.click(
            lambda: start_fresh_thread("New session"),
            outputs=fresh_outputs,
        )
        clear.click(on_clear, inputs=[confirm_clear], outputs=[rules, notice, confirm_clear])
    return demo


# ---------------------------------------------------------------------------
# Self-test (no Ollama)
# ---------------------------------------------------------------------------


def _scripted(tool_name: str, args: dict, call_id: str = "call-1"):
    def agent(state: GraphState) -> dict:
        if any(isinstance(message, ToolMessage) for message in state["messages"]):
            return {"messages": [AIMessage(content="final")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": args,
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    return agent


def _fresh(root: Path, name: str, store: PermissionStore, workspace: Path, tool: str, args: dict):
    saver = SqliteCheckpointSaver(root / f"{name}.sqlite")
    graph = build_graph(
        saver,
        store,
        agent_fn=_scripted(tool, args, call_id=name),
        workspace=workspace,
    )
    return saver, graph


def _self_test() -> int:
    root = _DATA_DIR / "_selftest"
    if root.exists():
        for child in root.rglob("*"):
            if child.is_file():
                child.unlink()
    workspace = ensure_workspace(root / "workspace")
    store = PermissionStore(root / "permissions.sqlite")
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    store.clear_rules()
    EXEC_LOG.clear()
    saver, graph = _fresh(root, "pause", store, workspace, "file_write", {"path": "note.txt", "content": "hello"})
    submit_task(graph, store, "t-pause", "write a file")
    pending = pending_payload(graph, "t-pause")
    note = workspace / "note.txt"
    check("1 pause before execution", bool(pending) and pending["tool"] == "file_write" and not note.exists())
    check("1 exact args shown", pending and pending["args"] == {"path": "note.txt", "content": "hello"})

    apply_decision(graph, "t-pause", pending["request_id"], "approve_once")
    check("2 approve once executes", note.read_text(encoding="utf-8") == "hello" if note.exists() else False)
    check("2 approve once saves no allow rule", store.lookup("file_write", "write") is None)
    check("10 single execution", EXEC_LOG.count("file_write") == 1)
    again = apply_decision(graph, "t-pause", pending["request_id"], "approve_once")
    check("10 duplicate resume ignored", "Duplicate" in again and EXEC_LOG.count("file_write") == 1)

    store.clear_rules()
    EXEC_LOG.clear()
    _, graph = _fresh(root, "allow", store, workspace, "file_write", {"path": "allow.txt", "content": "saved"})
    submit_task(graph, store, "t-allow", "write")
    pending = pending_payload(graph, "t-allow")
    apply_decision(graph, "t-allow", pending["request_id"], "approve_all")
    check("3 approve all saves rule", store.lookup("file_write", "write") == "allow")
    _, graph2 = _fresh(root, "allow2", store, workspace, "file_write", {"path": "allow2.txt", "content": "again"})
    submit_task(graph2, store, "t-allow-2", "write again")
    check("4 new session skips prompt", pending_payload(graph2, "t-allow-2") is None)
    check("4 second session executed", (workspace / "allow2.txt").exists())

    reopened = PermissionStore(root / "permissions.sqlite")
    check("5 rules survive new store connection", reopened.lookup("file_write", "write") == "allow")

    store.clear_rules()
    EXEC_LOG.clear()
    _, graph = _fresh(root, "deny", store, workspace, "shell", {"command": "pwd"})
    submit_task(graph, store, "t-deny", "pwd")
    pending = pending_payload(graph, "t-deny")
    apply_decision(graph, "t-deny", pending["request_id"], "deny_permanent")
    check("6 deny saved", store.lookup("shell", "read_only") == "deny")
    before = len(EXEC_LOG)
    _, graph = _fresh(root, "deny2", store, workspace, "shell", {"command": "pwd"})
    submit_task(graph, store, "t-deny-2", "pwd again")
    check("6 deny blocks next session", pending_payload(graph, "t-deny-2") is None and len(EXEC_LOG) == before)

    store.save_rule("shell", "read_only", "allow")
    check("7 deny beats allow", store.lookup("shell", "read_only") == "deny")
    EXEC_LOG.clear()
    _, graph = _fresh(root, "both", store, workspace, "shell", {"command": "ls"})
    submit_task(graph, store, "t-both", "ls")
    messages = graph.get_state(thread_config("t-both")).values["messages"]
    tool_msg = next(message for message in messages if isinstance(message, ToolMessage))
    body = json.loads(tool_msg.content)
    check("7 deny still blocks execution", body["executed"] is False and EXEC_LOG == [])

    EXEC_LOG.clear()
    store.clear_rules()
    _, graph = _fresh(root, "bad", store, workspace, "shell", {"command": "rm -rf /tmp/hitl-demo"})
    submit_task(graph, store, "t-bad", "restricted")
    pending = pending_payload(graph, "t-bad")
    check(
        "8 restricted shell pauses with the real command",
        bool(pending) and pending["args"].get("command") == "rm -rf /tmp/hitl-demo",
    )
    apply_decision(graph, "t-bad", pending["request_id"], "approve_all")
    messages = graph.get_state(thread_config("t-bad")).values["messages"]
    body = json.loads(next(message.content for message in messages if isinstance(message, ToolMessage)))
    check(
        "8 approval cannot execute a restricted command",
        body["decision"] == "allowlist" and body["executed"] is False and EXEC_LOG == [],
    )
    check("8 approve all did not save a shell allow rule", store.lookup("shell", "read_only") is None)
    _, graph = _fresh(root, "escape", store, workspace, "file_read", {"path": "../.env"})
    submit_task(graph, store, "t-escape", "read env")
    body = json.loads(
        next(
            message.content
            for message in graph.get_state(thread_config("t-escape")).values["messages"]
            if isinstance(message, ToolMessage)
        )
    )
    check("8 path traversal rejected", body["executed"] is False and "traversal" in body["reason"])

    _, graph = _fresh(root, "rej", store, workspace, "file_read", {"path": "welcome.txt"})
    submit_task(graph, store, "t-rej", "read")
    pending = pending_payload(graph, "t-rej")
    apply_decision(graph, "t-rej", pending["request_id"], "reject_once")
    body = json.loads(
        next(
            message.content
            for message in graph.get_state(thread_config("t-rej")).values["messages"]
            if isinstance(message, ToolMessage)
        )
    )
    check("9 structured rejection", body["status"] == "rejected" and body["executed"] is False)
    final = graph.get_state(thread_config("t-rej")).values["messages"][-1]
    check("9 agent continues after rejection", isinstance(final, AIMessage) and final.content == "final")

    store.clear_rules()
    EXEC_LOG.clear()
    saver, graph = _fresh(root, "restart", store, workspace, "file_write", {"path": "restart.txt", "content": "once"})
    submit_task(graph, store, "t-restart", "write")
    pending = pending_payload(graph, "t-restart")
    saver.close()
    saver2 = SqliteCheckpointSaver(root / "restart.sqlite")
    graph2 = build_graph(
        saver2,
        store,
        agent_fn=_scripted("file_write", {"path": "restart.txt", "content": "once"}, call_id="restart"),
        workspace=workspace,
    )
    restored = pending_payload(graph2, "t-restart")
    check("5 paused graph survives new checkpointer", bool(restored) and restored["request_id"] == pending["request_id"])
    apply_decision(graph2, "t-restart", restored["request_id"], "approve_once")
    check("10 resume executes once", EXEC_LOG.count("file_write") == 1)
    check("10 file written once", (workspace / "restart.txt").read_text(encoding="utf-8") == "once")

    rules_before = store.list_rules()
    new_id = "hitl-" + uuid.uuid4().hex[:12]
    check("11 new session id keeps rules", new_id != "t-restart" and store.list_rules() == rules_before)

    store.save_rule("file_read", "read", "allow")
    kept = clear_permission_rules(store, confirmed=False)
    check("12 clear without confirmation keeps rules", "not changed" in kept and store.lookup("file_read", "read") == "allow")
    cleared = clear_permission_rules(store, confirmed=True)
    check("12 confirmed clear removes rules", store.list_rules() == [] and "Removed" in cleared)
    store.try_begin("old-req", "shell")
    store.finish("old-req", '{"status":"ok","output":"OLD OUTPUT"}')
    clear_permission_rules(store, confirmed=True)
    check("12 clear drops cached tool output", store.get_done("old-req") is None)
    history = [
        HumanMessage(content="first"),
        AIMessage(content="OLD OUTPUT"),
        HumanMessage(content="again"),
    ]
    pending_view = {"tool": "shell"}
    chat = chat_from_messages(messages_for_display(history, pending_view), pending_view)
    blob = " ".join(item["content"] for item in chat)
    check("pending view hides previous answer", "OLD OUTPUT" not in blob and "again" in blob)

    store.close()
    reopened.close()
    saver2.close()
    print(f"{len(failures)} failed" if failures else "all self-tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    build_ui().launch()
