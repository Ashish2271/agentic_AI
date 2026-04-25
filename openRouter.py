#!/usr/bin/env python3
"""
Basic CLI chat with OpenRouter
Usage: python openrouter_chat.py
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
API_KEY   = "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"
BASE_URL  = "https://openrouter.ai/api/v1/chat/completions"
MODEL     = "openai/gpt-4o-mini"          # change to any OpenRouter model
SYSTEM    = "You are a helpful assistant."
# ─────────────────────────────────────────────────────────────────────────────


def chat(messages: list[dict]) -> Optional[str]:
    """Send messages to OpenRouter and return the assistant reply."""
    # Print messages before API call
    print("\n=== Messages being sent ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("===========================\n")

    
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
    }).encode()
  
    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer":  "https://localhost",   # required by OpenRouter
            "X-Title":       "CLI Chat",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"\n[HTTP {e.code}] {body}\n")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"\n[Parse error] {e}\n")
        return None


def main():
    if not API_KEY:
        print("Error: set the OPENROUTER_API_KEY environment variable first.")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        return

    print(f"OpenRouter Chat  •  model: {MODEL}")
    print("Type 'exit' or press Ctrl-C to quit.\n")

    history: list[dict] = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        reply = chat(history)

        if reply:
            print(reply)
            history.append({"role": "assistant", "content": reply})
        else:
            print("[no response — try again]")
            history.pop()   # drop the failed user message so history stays clean


if __name__ == "__main__":
    main()