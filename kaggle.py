# %% [markdown]
# ---
# title: "Can Language Models Learn a PDE?"
# description: >
#   Benchmark testing whether LLMs can identify unknown elliptic PDE coefficients
#   by probing the weak formulation with test functions.
# ---

# %% [markdown]
# ## Setup
#
# This notebook requires the PDE library files (pde.py, basis.py, weak_form.py,
# expression_parser.py, benchmark.py) to be available as a Kaggle Dataset or
# pasted into earlier cells.
#
# For local development, just ensure they're importable from the working directory.

# %%
from kaggle_benchmarks import assertions, chats, task
from kaggle_benchmarks.kaggle import model_proxy

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

# Your PDE library imports — adjust path as needed
from basis import LegendreBasis
from pde import EllipticPDE, EllipticSolution, solve_elliptic, make_random_elliptic_pde
from weak_form import _composite_simpson
from expression_parser import make_test_function_from_string
from benchmark import score_prediction, ScoreResult, DIFFICULTY_CONFIGS, BenchmarkTask

# Import the parsing and session logic from main_loop
from main_loop import (
    ProbeSession,
    parse_probe_response,
    format_score_result,
    protocol,
    DIFFICULTY_CONFIG,
    dispatch_turn,
)

# %% [markdown]
# ## Model Setup

# %%
# Define models to test — adjust to whatever's available on Kaggle
llm_gemini_flash = model_proxy.ModelProxy(model="google/gemini-2.5-flash", api="genai")
# llm_gemini_pro = model_proxy.ModelProxy(model="google/gemini-2.5-pro", api="genai")
# llm_claude_sonnet = model_proxy.ModelProxy(model="anthropic/claude-sonnet-4", api="openai")

# %% [markdown]
# ## Core Benchmark Loop
#
# This adapts `run_probe_session` from main_loop.py to use kbench's
# `llm.prompt()` instead of `backend.chat(system, messages)`.
#
# Key differences:
# - System prompt goes into `chats.new(system_instructions=...)`
# - `llm.prompt()` manages conversation history automatically
# - Tool dispatch logic is unchanged — we parse text, execute tools, feed results back

# %%
def build_system_prompt(session: ProbeSession, baseline: bool = False,
                        min_queries: int = None, max_turns: int = 36) -> str:
    """Build the full system prompt from session config."""
    full_system = session.system_prompt() + protocol
    full_system += """
After each COMPUTE: solve, report your confidence (0-100%) that \
each coefficient vector (a, b, c, f) is within 0.1 of the true \
value. Format exactly as:
  Confidence: a=XX%, b=XX%, c=XX%, f=XX%
Then give one sentence explaining the main source of your uncertainty.
"""
    if min_queries:
        full_system += (
            f"\nYou must submit at least {min_queries} queries before calling "
            f"COMPUTE: solve or PREDICT. You have {max_turns} turns total.\n"
        )
    return full_system


def build_initial_message(baseline: bool) -> str:
    """Build the first user message (with or without the ill-conditioning hint)."""
    base = (
        "Begin. Before each action, briefly explain your reasoning: "
        "what you expect to learn from it and why you chose it over alternatives."
    )
    if baseline:
        return base
    return base + (
        " Note: in elliptic inverse problems of this type, the three "
        "coefficient functions a(x), b(x), c(x) are not equally identifiable "
        "from weak-form data. The difficulty of recovering each coefficient "
        "depends on the magnitude of its contribution to the weak-form "
        "equations relative to the others — and this varies with the solution "
        "u(x) and the test functions you choose."
    )




# %% [markdown]
# ## Task Definition

# %%
@task("PDE Coefficient Identification")
def pde_identify(llm, difficulty: str = "medium", seed: int = 42,
                 baseline: bool = False) -> float:
    """
    Test whether an LLM can identify elliptic PDE coefficients via weak-form probing.

    Returns total coefficient error (lower is better).
    """
    # --- Setup ---
    dc = DIFFICULTY_CONFIG[difficulty]
    max_queries = dc["budget"]
    max_turns = max_queries * 2  # generous turn budget

    session = ProbeSession.from_difficulty(difficulty, seed, max_queries)
    system = build_system_prompt(session, baseline=baseline, max_turns=max_turns)
    initial_msg = build_initial_message(baseline)

    prev_solve_coeffs = {}
    run_log = {"turns": [], "solves": [], "queries": [], "verifications": [],
               "term_integrals": [], "error_curves": []}

    # --- Multi-turn probe loop ---
    with chats.new(system_instructions=system):
        # First turn: send the initial message, get first LLM response
        llm_text = llm.prompt(initial_msg)

        for turn in range(1, max_turns + 1):
            # Parse and dispatch
            actions = parse_probe_response(llm_text)

            run_log["turns"].append({
                "turn": turn,
                "timestamp": time.time(),
                "role": "assistant",
                "content": llm_text,
                "parsed_actions": [a["action"] for a in actions],
            })

            response_text, done, prev_solve_coeffs, _ = dispatch_turn(
                session, llm_text, actions, prev_solve_coeffs, run_log,
                turn, max_turns, verbose=True,
            )

            if done:
                break

            # Feed tool results back, get next LLM response
            llm_text = llm.prompt(response_text)

    # --- Auto-submit if needed ---
    if not session.prediction_submitted and session.last_solve_coeffs is not None:
        prediction = session.last_solve_coeffs
        session.submit_prediction(prediction)

    # --- Score ---
    if session.final_score:
        errors = session.final_score.to_dict()
    else:
        errors = {"a": 999, "b": 999, "c": 999, "f": 999, "total": 999}

    total_err = errors.get("total", errors.get("coefficient_errors", {}).get("total", 999))

    # --- Assertions ---
    assertions.assert_less_than(
        total_err, 10.0,
        expectation="Model should achieve some coefficient recovery (total error < 10)"
    )

    # --- Log detailed metrics (visible in run output) ---
    print(f"Difficulty: {difficulty}, Seed: {seed}, Baseline: {baseline}")
    print(f"Queries used: {session.queries_used}/{max_queries}")
    print(f"Turns used: {turn}")
    print(f"Coefficient errors: a={errors.get('a', '?'):.4f} b={errors.get('b', '?'):.4f} "
          f"c={errors.get('c', '?'):.4f} f={errors.get('f', '?'):.4f}")
    print(f"Total error: {total_err:.4f}")

    return total_err


# %% [markdown]
# ## Single Run (for testing)

# %%
result = pde_identify.run(
    llm=llm_gemini_flash,
    difficulty="easy",
    seed=42,
    baseline=False,
)

# %% [markdown]
# ## Sweep Across Seeds

# %%
import pandas as pd

eval_data = pd.DataFrame([
    {"difficulty": "medium", "seed": s, "baseline": b}
    for s in [1, 2, 3, 4, 5]
    for b in [False, True]
])

results = pde_identify.evaluate(llm=llm_gemini_flash, evaluation_data=eval_data)

# %% [markdown]
# ## Multi-Model Comparison

# %%
# models = [llm_gemini_flash, llm_gemini_pro, llm_claude_sonnet]
# results = pde_identify.evaluate(llm=models, evaluation_data=eval_data)

# %% [markdown]
# ## Submit to Leaderboard

# %%
# %choose PDE Coefficient Identification
