#!/usr/bin/env python3
"""
events.py — Lightweight EventBus for the OpenRouter agent.

Events
------
PreToolUse   fired before every tool call; a handler can cancel it
PostToolUse  fired after every tool call with the result
Stop         fired when the agent loop exits (normally or on error)

Usage
-----
    from events import bus, PreToolUse, PostToolUse, Stop

    @bus.on(PreToolUse)
    def my_hook(event: PreToolUse):
        print(f"about to run {event.tool_name}")

    # To block a tool call from a PreToolUse handler:
    #   event.cancel("reason")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


# ── Event types ───────────────────────────────────────────────────────────────

@dataclass
class PreToolUse:
    """
    Emitted before a tool is executed.
    Set ``event.cancelled = True`` (or call ``event.cancel()``) inside a
    handler to skip execution. The agent will use ``event.cancel_reason``
    as the tool result that gets sent back to the model.
    """
    tool_name:     str
    tool_call_id:  str
    arguments:     dict[str, Any]

    # mutable fields handlers may change
    cancelled:     bool = field(default=False, init=False)
    cancel_reason: str  = field(default="Tool call cancelled by handler.", init=False)

    def cancel(self, reason: str = "Tool call cancelled by handler.") -> None:
        self.cancelled     = True
        self.cancel_reason = reason


@dataclass
class PostToolUse:
    """Emitted after a tool finishes executing."""
    tool_name:    str
    tool_call_id: str
    arguments:    dict[str, Any]
    result:       str               # the string returned by execute_tool


@dataclass
class Stop:
    """
    Emitted when the agent loop exits for the current turn.
    ``reason`` is one of: 'stop', 'max_rounds', 'error', 'cancelled'
    ``final_reply`` is the last assistant text (may be empty on error).
    """
    reason:      str
    final_reply: str
    error:       str | None = None


# ── EventBus ──────────────────────────────────────────────────────────────────

EventType = type  # just an alias for readability

class EventBus:
    """
    Simple synchronous publish/subscribe bus.

    Register:   bus.on(EventClass)(handler_fn)   or use @bus.on(EventClass)
    Emit:       bus.emit(event_instance)
    Remove:     bus.off(EventClass, handler_fn)
    """

    def __init__(self) -> None:
        self._listeners: dict[EventType, list[Callable]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def on(self, event_type: EventType) -> Callable:
        """Decorator / function that registers a handler for an event type."""
        def decorator(fn: Callable) -> Callable:
            self._listeners.setdefault(event_type, []).append(fn)
            return fn
        return decorator

    def off(self, event_type: EventType, fn: Callable) -> None:
        """Unregister a previously registered handler."""
        handlers = self._listeners.get(event_type, [])
        if fn in handlers:
            handlers.remove(fn)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def emit(self, event: Any) -> Any:
        """
        Call every registered handler for the event's type, in registration order.
        Returns the (possibly mutated) event so callers can inspect it.
        """
        for handler in self._listeners.get(type(event), []):
            handler(event)
        return event


# ── Global singleton ──────────────────────────────────────────────────────────

bus = EventBus()


# ── Built-in approval handler (interactive) ───────────────────────────────────

# ANSI helpers (duplicated here so events.py stays standalone)
_R = "\033[0m"; _Y = "\033[33m"; _B = "\033[1m"; _D = "\033[2m"; _RE = "\033[31m"

def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())


def approval_handler(event: PreToolUse) -> None:
    """
    Interactive approval gate.  Registered with the bus by default.
    Prints the pending tool call and asks the user to approve/deny.

    Set ``AGENT_AUTO_APPROVE=1`` in the environment to skip prompts.
    """
    if os.environ.get("AGENT_AUTO_APPROVE") == "1":
        return   # skip prompts when running non-interactively

    print(
        f"\n{_Y}  ┌─ Tool request ────────────────────────────────{_R}\n"
        f"{_Y}  │{_R}  {_B}{event.tool_name}{_R}({_D}{_fmt_args(event.arguments)}{_R})\n"
        f"{_Y}  └───────────────────────────────────────────────{_R}"
    )
    try:
        answer = input("  Approve? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("n", "no"):
        reason = input("  Reason (optional): ").strip() or "Denied by user."
        event.cancel(reason)
        print(f"  {_RE}✗ Cancelled:{_R} {reason}")
    else:
        print(f"  ✓ Approved")


import os   # needed by approval_handler; imported here to keep grouping clean
bus.on(PreToolUse)(approval_handler)