#!/usr/bin/env python3
"""
council.py — AI Council built on agent.py's backbone.

Reuses call_api(), run_agent(), Session, and the EventBus from agent.py.
Calls multiple models in parallel, then has a judge LLM pick the winner.

Usage (CLI):
    python council.py "What is the best sorting algorithm for nearly-sorted data?"
    python council.py "Explain RAFT consensus" --models gpt-4o claude-sonnet gemini-pro
    python council.py --list-models

Usage (library):
    from council import ai_council
    result = ai_council("Explain the CAP theorem")
    print(result.best_response)
    print(result.summary())

Env:
    OPENROUTER_API_KEY=sk-or-...
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# ── Reuse agent.py backbone ───────────────────────────────────────────────────
# run_agent() gives us the full tool-calling loop, EventBus hooks, and Session.
# call_api() is the raw HTTP wrapper we also use for the judge.
import agent as _agent_module
from agent import call_api, run_agent, SYSTEM, c
from history import Session
from models import AVAILABLE_MODELS

load_dotenv()

# ── Council model roster ──────────────────────────────────────────────────────
# Mirrors whatever is in models.py so adding a model there adds it here too.
COUNCIL_MODELS: dict[str, str] = dict(AVAILABLE_MODELS)

# Default council: first 3 entries in models.py
DEFAULT_COUNCIL: list[str] = list(AVAILABLE_MODELS.keys())[:3]

# Judge model: first entry in models.py (swap to something smarter if you like)
JUDGE_MODEL_KEY: str = list(AVAILABLE_MODELS.keys())[0]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    model_key:  str
    model_id:   str
    response:   str
    elapsed_ms: int
    error:      Optional[str] = None
    turns:      int = 0
    tool_calls: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RatedResponse:
    model_response: ModelResponse
    scores:         dict[str, int]   # criterion → 1-10
    reasoning:      str
    total:          int = field(init=False)

    def __post_init__(self):
        self.total = sum(self.scores.values())


@dataclass
class CouncilResult:
    question:   str
    responses:  list[ModelResponse]
    ratings:    list[RatedResponse]
    winner:     RatedResponse
    elapsed_ms: int

    @property
    def best_response(self) -> str:
        return self.winner.model_response.response

    def summary(self) -> str:
        lines = [
            f"Question : {self.question[:80]}{'...' if len(self.question) > 80 else ''}",
            f"Council  : {', '.join(r.model_response.model_key for r in self.ratings)}",
            f"Winner   : {self.winner.model_response.model_key} "
            f"(score {self.winner.total}/40)",
            "",
            "Scores:",
        ]
        for rated in sorted(self.ratings, key=lambda r: r.total, reverse=True):
            bar = "█" * (rated.total // 4) + "░" * (10 - rated.total // 4)
            lines.append(
                f"  {rated.model_response.model_key:<22} [{bar}] {rated.total:>3}/40  "
                + ", ".join(f"{k[0].upper()}:{v}" for k, v in rated.scores.items())
            )
        lines += [
            "",
            f"Best response ({self.winner.model_response.model_key}):",
            self.winner.model_response.response,
        ]
        return "\n".join(lines)


# ── Single model worker ───────────────────────────────────────────────────────

def _run_one_model(model_key: str, model_id: str, question: str) -> ModelResponse:
    """
    Run one council member using agent.py's full run_agent() loop.
    Temporarily swaps the global MODEL, then restores it.
    Thread-safe: each call gets its own history list and Session.
    """
    t0 = time.monotonic()
    saved_model = _agent_module.MODEL
    try:
        _agent_module.MODEL = model_id

        session = Session.start(model=model_id)
        history: list[dict] = [{"role": "system", "content": SYSTEM}]
        session.write_system(SYSTEM)
        history.append({"role": "user", "content": question})
        session.write_user(question)

        stats = {"turns": 0, "tool_calls": 0}
        reply = run_agent(history, session, stats)
        session.end(turns=stats["turns"], tool_calls=stats["tool_calls"])

        return ModelResponse(
            model_key=model_key,
            model_id=model_id,
            response=reply,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            turns=stats["turns"],
            tool_calls=stats["tool_calls"],
        )
    except Exception as e:
        return ModelResponse(
            model_key=model_key,
            model_id=model_id,
            response="",
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=str(e),
        )
    finally:
        _agent_module.MODEL = saved_model


# ── Judge ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a rigorous, impartial judge evaluating AI responses. "
    "Score each response on four criteria (1–10 each): "
    "accuracy (factual correctness and completeness), "
    "clarity (structure and ease of understanding), "
    "conciseness (appropriately brief without losing depth), "
    "usefulness (practical value and actionability). "
    "Return ONLY valid JSON — no markdown fences, no preamble, no commentary. "
    "Schema: "
    '{"ratings": [{"model_key": "<key>", '
    '"scores": {"accuracy": 8, "clarity": 7, "conciseness": 6, "usefulness": 8}, '
    '"reasoning": "One sentence."}]}'
)


def _judge(question: str, responses: list[ModelResponse]) -> list[RatedResponse]:
    """Call the judge model (via call_api) and parse its ratings."""
    saved_model = _agent_module.MODEL
    try:
        _agent_module.MODEL = COUNCIL_MODELS[JUDGE_MODEL_KEY]

        blocks = [f"=== MODEL: {r.model_key} ===\n{r.response}" for r in responses]
        prompt = (
            f"QUESTION:\n{question}\n\n"
            "RESPONSES TO RATE:\n\n"
            + "\n\n".join(blocks)
            + "\n\nRate every response listed above. Return JSON only."
        )

        raw_resp = call_api([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ])
        raw_text = raw_resp["choices"][0]["message"]["content"] or ""

        # Strip markdown fences if the model adds them despite instructions
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:]).rstrip("`").strip()

        data = json.loads(clean)

    except Exception:
        # Fallback: equal scores so the run doesn't crash
        return [
            RatedResponse(
                model_response=r,
                scores={"accuracy": 5, "clarity": 5, "conciseness": 5, "usefulness": 5},
                reasoning="Judge failed — equal fallback scores applied.",
            )
            for r in responses
        ]
    finally:
        _agent_module.MODEL = saved_model

    resp_by_key = {r.model_key: r for r in responses}
    rated: list[RatedResponse] = []
    rated_keys: set[str] = set()

    for item in data.get("ratings", []):
        key = item.get("model_key", "")
        if key not in resp_by_key:
            continue
        rated.append(RatedResponse(
            model_response=resp_by_key[key],
            scores=item.get("scores", {}),
            reasoning=item.get("reasoning", ""),
        ))
        rated_keys.add(key)

    # Ensure every response has an entry (zeroed if judge missed it)
    for r in responses:
        if r.model_key not in rated_keys:
            rated.append(RatedResponse(
                model_response=r,
                scores={"accuracy": 0, "clarity": 0, "conciseness": 0, "usefulness": 0},
                reasoning="Not rated by judge.",
            ))
    return rated


# ── Public API ────────────────────────────────────────────────────────────────

def ai_council(
    question:    str,
    models:      Optional[list[str]] = None,
    max_workers: int = 6,
) -> CouncilResult:
    """
    Convene the AI Council.

    Steps:
      1. Call each model via agent.py's run_agent() in parallel threads.
         Each model gets the full tool-calling loop, EventBus hooks, and its
         own Session written to history/.
      2. Send all responses to a judge LLM (via call_api) for scoring.
      3. Return CouncilResult with .best_response and .summary().

    Args:
        question:    The prompt to put to the council.
        models:      List of model keys from models.py / COUNCIL_MODELS.
                     Defaults to DEFAULT_COUNCIL (first 3 in models.py).
        max_workers: Thread pool size for parallel model calls.
    """
    model_keys = models or DEFAULT_COUNCIL
    unknown = [k for k in model_keys if k not in COUNCIL_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {unknown}. "
            f"Available: {list(COUNCIL_MODELS.keys())}"
        )

    t0 = time.monotonic()

    # 1. Run all models in parallel, each through run_agent()
    responses: list[ModelResponse] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_model, key, COUNCIL_MODELS[key], question): key
            for key in model_keys
        }
        for future in as_completed(futures):
            responses.append(future.result())

    ok = [r for r in responses if r.ok]
    if not ok:
        errors = "\n".join(f"  {r.model_key}: {r.error}" for r in responses)
        raise RuntimeError(f"All council members failed:\n{errors}")

    # 2. Judge
    ratings = _judge(question, ok)
    winner  = max(ratings, key=lambda rr: rr.total)

    return CouncilResult(
        question=question,
        responses=responses,
        ratings=ratings,
        winner=winner,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(result: CouncilResult) -> None:
    print()
    print(c("bold",  "━━━ AI COUNCIL RESULTS ━━━"))
    print(c("dim",   f"Question   : {result.question[:100]}"))
    print(c("dim",   f"Council    : {', '.join(r.model_response.model_key for r in result.ratings)}"))
    print(c("dim",   f"Judge      : {JUDGE_MODEL_KEY}"))
    print(c("dim",   f"Total time : {result.elapsed_ms}ms"))
    print()

    sorted_ratings = sorted(result.ratings, key=lambda r: r.total, reverse=True)
    print(c("bold", "Leaderboard:"))
    for rated in sorted_ratings:
        is_winner = rated.model_response.model_key == result.winner.model_response.model_key
        crown     = " 👑" if is_winner else "   "
        bar_fill  = rated.total // 4
        bar       = "█" * bar_fill + "░" * (10 - bar_fill)
        score_str = f"{rated.total}/40"
        detail    = "  " + " ".join(f"{k[0].upper()}:{v}" for k, v in rated.scores.items())
        latency   = f"  {rated.model_response.elapsed_ms}ms"

        row = (
            f"  {rated.model_response.model_key:<22}"
            f" [{bar}] {score_str:<8}"
            + c("dim", detail + latency)
            + crown
        )
        print(c("green" if is_winner else "reset", row))
        if rated.reasoning:
            print(c("dim", f"    ↳ {rated.reasoning}"))

    print()
    print(c("bold", f"Best Response  [{result.winner.model_response.model_key}]:"))
    print(c("cyan", "─" * 60))
    print(result.best_response)
    print(c("cyan", "─" * 60))

    failed = [r for r in result.responses if not r.ok]
    if failed:
        print()
        print(c("red", "Failed models:"))
        for r in failed:
            print(c("red", f"  ✗ {r.model_key}: {r.error}"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AI Council — multi-model answer competition powered by agent.py"
    )
    parser.add_argument("question", nargs="?", help="Question to ask the council")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(COUNCIL_MODELS.keys()),
        default=None,
        help=f"Models to convene (default: {DEFAULT_COUNCIL})",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print(c("bold", "Available council models (from models.py):"))
        for key, mid in COUNCIL_MODELS.items():
            tags = []
            if key in DEFAULT_COUNCIL:
                tags.append(c("green", "default council"))
            if key == JUDGE_MODEL_KEY:
                tags.append(c("yellow", "judge"))
            tag_str = "  ← " + ", ".join(tags) if tags else ""
            print(f"  {key:<22} → {mid}{tag_str}")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        print(c("red", "Error: OPENROUTER_API_KEY is not set."))
        sys.exit(1)

    question = args.question
    if not question:
        try:
            print(c("cyan", "Question: "), end="", flush=True)
            question = input().strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    if not question:
        print(c("red", "No question provided."))
        sys.exit(1)

    models = args.models or DEFAULT_COUNCIL
    print(c("dim", f"\nConvening council: {', '.join(models)}"))
    print(c("dim",  "Running each model through agent.py's tool loop in parallel...\n"))

    try:
        result = ai_council(question, models=models)
        print_result(result)
    except KeyboardInterrupt:
        print(c("dim", "\nCancelled."))
    except Exception as e:
        print(c("red", f"\nError: {e}"))
        sys.exit(1)


if __name__ == "__main__":
    main()