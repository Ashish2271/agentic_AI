#!/usr/bin/env python3
"""
history.py — Persists conversation history to disk, one session per folder.

Folder layout
-------------
history/
└── 2025-01-30_14-22-05_a3f2/          ← session folder  (timestamp + 4-char uid)
    ├── messages.jsonl                   ← one JSON object per line, appended live
    └── session.json                     ← written once at session end

messages.jsonl format (one object per line)
-------------------------------------------
Every entry has at minimum:
  { "ts": "14:22:05", "role": "...", ... }

Roles written:
  system      — the system prompt (written once at start)
  user        — human turn
  assistant   — model reply (text only; tool_calls stripped for readability)
  tool        — tool result
  tool_call   — synthetic record written by the agent before execution
  event       — lifecycle event  (PreToolUse / PostToolUse / Stop)

session.json format
-------------------
  {
    "session_id":  "2025-01-30_14-22-05_a3f2",
    "started_at":  "2025-01-30T14:22:05",
    "ended_at":    "2025-01-30T14:35:11",
    "model":       "openai/gpt-4o-mini",
    "turns":       7,
    "tool_calls":  3,
    "messages":    42
  }
"""

from __future__ import annotations

import json
import os
import uuid
import datetime
from pathlib import Path
from typing  import Any


HISTORY_ROOT = Path("history")


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """
    Manages one agent session on disk.

    Usage
    -----
        session = Session.start(model="openai/gpt-4o-mini")
        session.write_system("You are …")
        session.write_user("Hello")
        session.write_assistant("Hi!")
        session.write_tool_call("read_file", {"path": "x.py"}, "call-id-123")
        session.write_tool_result("read_file", "call-id-123", "file contents…")
        session.write_event("Stop", {"reason": "stop"})
        session.end(turns=1, tool_calls=1)
    """

    def __init__(self, session_dir: Path, model: str) -> None:
        self._dir        = session_dir
        self._model      = model
        self._jsonl      = session_dir / "messages.jsonl"
        self._started_at = datetime.datetime.now()
        self._count      = 0   # messages written

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def start(cls, model: str = "") -> "Session":
        """Create a new session folder and return the Session object."""
        ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        uid = uuid.uuid4().hex[:4]
        sid = f"{ts}_{uid}"
        session_dir = HISTORY_ROOT / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        return cls(session_dir, model)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._dir.name

    @property
    def path(self) -> Path:
        return self._dir

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _append(self, record: dict[str, Any]) -> None:
        """Append one JSON record to messages.jsonl."""
        record.setdefault("ts", datetime.datetime.now().strftime("%H:%M:%S"))
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1

    def write_system(self, content: str) -> None:
        self._append({"role": "system", "content": content})

    def write_user(self, content: str) -> None:
        self._append({"role": "user", "content": content})

    def write_assistant(self, content: str) -> None:
        self._append({"role": "assistant", "content": content})

    def write_tool_call(self, tool_name: str, arguments: dict, tool_call_id: str) -> None:
        """Written just before a tool executes (or is cancelled)."""
        self._append({
            "role":         "tool_call",
            "tool_name":    tool_name,
            "tool_call_id": tool_call_id,
            "arguments":    arguments,
        })

    def write_tool_result(self, tool_name: str, tool_call_id: str, result: str) -> None:
        self._append({
            "role":         "tool",
            "tool_name":    tool_name,
            "tool_call_id": tool_call_id,
            "content":      result,
        })

    def write_event(self, event_name: str, data: dict[str, Any]) -> None:
        self._append({"role": "event", "event": event_name, **data})

    # ── Session summary ───────────────────────────────────────────────────────

    def end(self, turns: int = 0, tool_calls: int = 0) -> None:
        """Write session.json summary. Call once when the agent exits."""
        summary = {
            "session_id":  self.session_id,
            "started_at":  self._started_at.isoformat(timespec="seconds"),
            "ended_at":    datetime.datetime.now().isoformat(timespec="seconds"),
            "model":       self._model,
            "turns":       turns,
            "tool_calls":  tool_calls,
            "messages":    self._count,
        }
        summary_path = self._dir / "session.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


# ── Convenience reader ────────────────────────────────────────────────────────

def list_sessions() -> list[Path]:
    """Return all session folders sorted newest-first."""
    if not HISTORY_ROOT.exists():
        return []
    return sorted(HISTORY_ROOT.iterdir(), reverse=True)


def load_messages(session_id: str) -> list[dict]:
    """Load all messages from a session's messages.jsonl."""
    path = HISTORY_ROOT / session_id / "messages.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]