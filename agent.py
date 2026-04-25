#!/usr/bin/env python3
"""
agent.py — OpenRouter agent with tool-calling loop.
Imports tool definitions and implementations from tools.py.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agent.py
"""

import json
import os
import urllib.request
import urllib.error

from tools import TOOL_DEFINITIONS, execute_tool

# ── Config ────────────────────────────────────────────────────────────────────
# API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL    = "openai/gpt-4o-mini"   # swap to any OpenRouter model that supports tools
SYSTEM   = (
    "You are a helpful coding agent. "
    "Use the provided tools whenever you need to read/write files, "
    "run commands, or search the filesystem. "
    "Always reason step-by-step before acting."
)
MAX_TOOL_ROUNDS = 10   # safety limit — stops infinite tool loops
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
    """Send messages + tools to OpenRouter. Returns the raw response dict."""

        # Print messages before API call
    print("\n=== Messages being sent ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("===========================\n")



    payload = json.dumps({
        "model":    MODEL,
        "messages": messages,
        "tools":    TOOL_DEFINITIONS,
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
    Run the agent loop for the current turn.
    Keeps calling the model and executing tools until finish_reason == 'stop'.
    Returns the final assistant text.
    """
    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = call_api(history)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return f"[HTTP {e.code}] {body}"

        choice  = response["choices"][0]
        message = choice["message"]
        reason  = choice.get("finish_reason", "stop")

        # Always add the raw assistant message to history
        history.append(message)

        # ── No tool calls → we're done ────────────────────────────────────
        if reason == "stop" or not message.get("tool_calls"):
            return message.get("content") or ""

        # ── Execute every tool call the model requested ───────────────────
        for tool_call in message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            print(c("yellow", f"\n  ⚙  {fn_name}"), c("dim", json.dumps(args)))

            result = execute_tool(fn_name, args)

            print(c("dim", f"     → {result[:200]}{'…' if len(result) > 200 else ''}"))

            # Append the tool result so the model can read it next round
            history.append({
                "role":         "tool",
                "tool_call_id": tool_call["id"],
                "name":         fn_name,
                "content":      result,
            })

    return "[max tool rounds reached]"


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print(c("red", "Error: OPENROUTER_API_KEY is not set."))
        print("  export OPENROUTER_API_KEY=sk-or-...")
        return

    print(c("bold", f"\nOpenRouter Agent  •  model: {MODEL}"))
    print(c("dim",  f"Tools: {', '.join(t['function']['name'] for t in TOOL_DEFINITIONS)}"))
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

        # history already updated inside run_agent; just ensure final reply is stored
        if history[-1].get("role") != "assistant" or history[-1].get("content") != reply:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()