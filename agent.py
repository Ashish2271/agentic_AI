#!/usr/bin/env python3
"""
agent.py — OpenRouter agent with tool-calling loop, EventBus, and session history.

Folder layout written each run
-------------------------------
history/
└── 2025-01-30_14-22-05_a3f2/
    ├── messages.jsonl   ← every message appended live
    └── session.json     ← summary written on exit

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agent.py

Env flags:
    AGENT_AUTO_APPROVE=1   skip interactive tool-approval prompts
"""

import json
import os
import urllib.request
import urllib.error
from dotenv import load_dotenv
import argparse
from tools   import TOOL_DEFINITIONS, execute_tool
from events  import bus, PreToolUse, PostToolUse, Stop
from history import Session
from models import AVAILABLE_MODELS
load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = AVAILABLE_MODELS["gpt-4o-mini"]
SYSTEM   = (
    "You are a helpful coding agent. "
    "Use the provided tools whenever you need to read/write files, "
    "run commands, or search the filesystem. "
    "Always reason step-by-step before acting."
)
MAX_TOOL_ROUNDS = 10
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[36m",
    "yellow": "\033[33m",
    "green":  "\033[32m",
    "red":    "\033[31m",
    "dim":    "\033[2m",
}

def c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(messages: list[dict]) -> dict:
    payload = json.dumps({
        "model":       MODEL,
        "messages":    messages,
        "tools":       TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }).encode()

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer":  "https://localhost",
            "X-Title":       "CLI Agent",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(history: list[dict], session: Session, stats: dict) -> str:
    """
    Agentic loop: call model → handle tool calls → repeat until stop.
    All activity is recorded to `session` live.

    `stats` is a mutable dict with keys: turns, tool_calls
    """

    for _round in range(MAX_TOOL_ROUNDS):

        # ── 1. Call the model ─────────────────────────────────────────────
        try:
            response = call_api(history)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            error_msg  = f"[HTTP {e.code}] {error_body}"
            session.write_event("Stop", {"reason": "error", "error": error_msg})
            bus.emit(Stop(reason="error", final_reply="", error=error_msg))
            return error_msg

        choice  = response["choices"][0]
        message = choice["message"]
        reason  = choice.get("finish_reason", "stop")

        history.append(message)

        # ── 2. No tool calls → natural stop ──────────────────────────────
        if reason == "stop" or not message.get("tool_calls"):
            final = message.get("content") or ""
            session.write_assistant(final)
            session.write_event("Stop", {"reason": "stop"})
            bus.emit(Stop(reason="stop", final_reply=final))
            stats["turns"] += 1
            return final

        # ── 3. Process each tool call ─────────────────────────────────────
        # Write the assistant's text part (if any) before tool calls
        if message.get("content"):
            session.write_assistant(message["content"])

        for tool_call in message["tool_calls"]:
            fn_name      = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]

            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            # Record the pending call
            session.write_tool_call(fn_name, args, tool_call_id)

            # ── PreToolUse ────────────────────────────────────────────────
            pre = bus.emit(PreToolUse(
                tool_name    = fn_name,
                tool_call_id = tool_call_id,
                arguments    = args,
            ))
            session.write_event("PreToolUse", {
                "tool_name":  fn_name,
                "cancelled":  pre.cancelled,
                "cancel_reason": pre.cancel_reason if pre.cancelled else None,
            })

            if pre.cancelled:
                result = f"[Cancelled] {pre.cancel_reason}"
                print(c("red", f"  ✗ {fn_name} cancelled: {pre.cancel_reason}"))
            else:
                # ── Execute ───────────────────────────────────────────────
                print(c("yellow", f"\n  ⚙  {fn_name}"), c("dim", json.dumps(args)))
                result = execute_tool(fn_name, args)
                preview = result[:200] + ("…" if len(result) > 200 else "")
                print(c("dim", f"     → {preview}"))
                stats["tool_calls"] += 1

            # Record the result
            session.write_tool_result(fn_name, tool_call_id, result)

            # ── PostToolUse ───────────────────────────────────────────────
            bus.emit(PostToolUse(
                tool_name    = fn_name,
                tool_call_id = tool_call_id,
                arguments    = args,
                result       = result,
            ))
            session.write_event("PostToolUse", {
                "tool_name": fn_name,
                "result_preview": result[:200],
            })

            # Feed result back into history
            history.append({
                "role":         "tool",
                "tool_call_id": tool_call_id,
                "name":         fn_name,
                "content":      result,
            })

    # ── Hit round limit ───────────────────────────────────────────────────────
    msg = "[max tool rounds reached]"
    session.write_event("Stop", {"reason": "max_rounds"})
    bus.emit(Stop(reason="max_rounds", final_reply=msg))
    return msg


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print(c("red", "Error: OPENROUTER_API_KEY is not set."))
        print("  export OPENROUTER_API_KEY=sk-or-...")
        return
 # Argument parser for model selection
    parser = argparse.ArgumentParser(description='Run the OpenRouter agent.')
    parser.add_argument('--model', choices=AVAILABLE_MODELS.keys(), default='gpt-4o-mini',
                        help='Choose the model to use.')
    args = parser.parse_args()
    
    # Set the model based on the command-line argument
    global MODEL
    MODEL = AVAILABLE_MODELS[args.model]
    auto = os.environ.get("AGENT_AUTO_APPROVE") == "1"

    # ── Start session ─────────────────────────────────────────────────────
    session = Session.start(model=MODEL)

    print(c("bold",  f"\nOpenRouter Agent  •  model: {MODEL}"))
    print(c("dim",   f"Tools: {', '.join(t['function']['name'] for t in TOOL_DEFINITIONS)}"))
    print(c("dim",   f"Approval: {'auto' if auto else 'interactive'}"))
    print(c("purple" if hasattr(c, '__call__') else "dim",
            f"Session: {session.session_id}  →  {session.path}"))
    print(c("dim",   "Type 'exit' to quit.\n"))

    history: list[dict] = [{"role": "system", "content": SYSTEM}]
    session.write_system(SYSTEM)

    stats = {"turns": 0, "tool_calls": 0}

    try:
        while True:
            try:
                user_input = input(c("cyan", "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("dim", "\nBye!"))
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print(c("dim", "Bye!"))
                break

            session.write_user(user_input)
            history.append({"role": "user", "content": user_input})

            print(c("green", "Agent:"), end=" ", flush=True)
            reply = run_agent(history, session, stats)
            print(reply)

            if history[-1].get("role") != "assistant":
                history.append({"role": "assistant", "content": reply})

    finally:
        # Always write the summary, even if we crash or Ctrl-C
        session.end(turns=stats["turns"], tool_calls=stats["tool_calls"])
        print(c("dim", f"\nHistory saved → {session.path}"))


if __name__ == "__main__":
    main()