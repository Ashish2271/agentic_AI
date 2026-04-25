#!/usr/bin/env python3
"""
agent.py — OpenRouter agent with tool-calling loop + EventBus.

Events fired each turn
-----------------------
  PreToolUse   → before a tool runs  (handler may cancel it)
  PostToolUse  → after  a tool runs  (carries the result)
  Stop         → when the loop exits (reason + final reply)

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agent.py

Set AGENT_AUTO_APPROVE=1 to skip interactive approval prompts.
"""

import json
import os
import urllib.request
import urllib.error

from tools  import TOOL_DEFINITIONS, execute_tool
from events import bus, PreToolUse, PostToolUse, Stop

# ── Config ────────────────────────────────────────────────────────────────────
# API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL    = "openai/gpt-4o-mini"
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

def run_agent(history: list[dict]) -> str:
    """
    Agentic loop: call model → handle tool calls → repeat until stop.

    Events emitted
    --------------
    PreToolUse   before each tool call  (cancellable)
    PostToolUse  after  each tool call
    Stop         when the loop exits
    """

    for _round in range(MAX_TOOL_ROUNDS):

        # ── 1. Call the model ─────────────────────────────────────────────
        try:
            response = call_api(history)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            error_msg  = f"[HTTP {e.code}] {error_body}"
            bus.emit(Stop(reason="error", final_reply="", error=error_msg))
            return error_msg

        choice  = response["choices"][0]
        message = choice["message"]
        reason  = choice.get("finish_reason", "stop")

        history.append(message)

        # ── 2. No tool calls → natural stop ──────────────────────────────
        if reason == "stop" or not message.get("tool_calls"):
            final = message.get("content") or ""
            bus.emit(Stop(reason="stop", final_reply=final))
            return final

        # ── 3. Process each tool call ─────────────────────────────────────
        for tool_call in message["tool_calls"]:
            fn_name      = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]

            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            # ── PreToolUse ────────────────────────────────────────────────
            pre = bus.emit(PreToolUse(
                tool_name    = fn_name,
                tool_call_id = tool_call_id,
                arguments    = args,
            ))

            if pre.cancelled:
                result = f"[Cancelled] {pre.cancel_reason}"
                print(c("red", f"  ✗ {fn_name} cancelled: {pre.cancel_reason}"))
            else:
                # ── Execute ───────────────────────────────────────────────
                print(c("yellow", f"\n  ⚙  {fn_name}"), c("dim", json.dumps(args)))
                result = execute_tool(fn_name, args)
                preview = result[:200] + ("…" if len(result) > 200 else "")
                print(c("dim", f"     → {preview}"))

            # ── PostToolUse ───────────────────────────────────────────────
            bus.emit(PostToolUse(
                tool_name    = fn_name,
                tool_call_id = tool_call_id,
                arguments    = args,
                result       = result,
            ))

            # Feed result back into history for the next model call
            history.append({
                "role":         "tool",
                "tool_call_id": tool_call_id,
                "name":         fn_name,
                "content":      result,
            })

    # ── Hit round limit ───────────────────────────────────────────────────────
    msg = "[max tool rounds reached]"
    bus.emit(Stop(reason="max_rounds", final_reply=msg))
    return msg


# ── Example extra listeners (opt-in) ─────────────────────────────────────────
# Uncomment to log every tool call + result to a file.
#
# import datetime
# LOG_FILE = "agent.log"
#
# @bus.on(PreToolUse)
# def log_pre(event: PreToolUse):
#     with open(LOG_FILE, "a") as f:
#         f.write(f"[{datetime.datetime.now():%H:%M:%S}] PRE  {event.tool_name} {event.arguments}\n")
#
# @bus.on(PostToolUse)
# def log_post(event: PostToolUse):
#     with open(LOG_FILE, "a") as f:
#         f.write(f"[{datetime.datetime.now():%H:%M:%S}] POST {event.tool_name} → {event.result[:120]}\n")
#
# @bus.on(Stop)
# def log_stop(event: Stop):
#     with open(LOG_FILE, "a") as f:
#         f.write(f"[{datetime.datetime.now():%H:%M:%S}] STOP reason={event.reason}\n\n")


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print(c("red", "Error: OPENROUTER_API_KEY is not set."))
        print("  export OPENROUTER_API_KEY=sk-or-...")
        return

    auto = os.environ.get("AGENT_AUTO_APPROVE") == "1"

    print(c("bold", f"\nOpenRouter Agent  •  model: {MODEL}"))
    print(c("dim",  f"Tools: {', '.join(t['function']['name'] for t in TOOL_DEFINITIONS)}"))
    print(c("dim",  f"Approval: {'auto' if auto else 'interactive (set AGENT_AUTO_APPROVE=1 to skip)'}"))
    print(c("dim",  "Type 'exit' to quit.\n"))

    history: list[dict] = [{"role": "system", "content": SYSTEM}]

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

        history.append({"role": "user", "content": user_input})

        print(c("green", "Agent:"), end=" ", flush=True)
        reply = run_agent(history)
        print(reply)

        if history[-1].get("role") != "assistant":
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()