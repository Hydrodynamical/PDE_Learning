"""
LLM loop for interactive PDE identification.

Connects an LLM (via Anthropic API) to an InteractiveSession.
The LLM reads the problem, submits test functions, and eventually
predicts the PDE coefficients.

Protocol:
    - LLM writes QUERY: <expression>     to probe with a test function
    - LLM writes PREDICT:                to submit a final answer
        a_coeffs = [...]
        b_coeffs = [...]
        c_coeffs = [...]
        f_coeffs = [...]
    - Everything else is treated as reasoning (kept in conversation but not acted on)

Usage:
    python llm_loop.py                          # run with Anthropic API
    python llm_loop.py --mock                   # run with mock LLM (for testing)
    python llm_loop.py --difficulty hard --seed 123 --max-queries 30
"""

from __future__ import annotations
import re
import json
import argparse
import time
from typing import Optional

from interactive import InteractiveSession


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> list[dict]:
    """
    Parse an LLM response into a list of actions.

    Returns a list of dicts, each one of:
        {"action": "query", "spec": "..."}
        {"action": "decompose", "spec": "..."}
        {"action": "compute", "command": "..."}
        {"action": "predict", "a": [...], "b": [...], "c": [...], "f": [...]}
        {"action": "reasoning", "text": "..."}

    The LLM may include multiple actions in one response.
    """
    actions = []
    lines = text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check for QUERY:
        m = re.match(r'^QUERY:\s*(.+)$', line, re.IGNORECASE)
        if m:
            spec = m.group(1).strip().strip('"').strip("'").strip('`')
            actions.append({"action": "query", "spec": spec})
            i += 1
            continue

        # Check for DECOMPOSE:
        m = re.match(r'^DECOMPOSE:\s*(.+)$', line, re.IGNORECASE)
        if m:
            spec = m.group(1).strip().strip('"').strip("'").strip('`')
            actions.append({"action": "decompose", "spec": spec})
            i += 1
            continue

        # Check for COMPUTE:
        m = re.match(r'^COMPUTE:\s*(.+)$', line, re.IGNORECASE)
        if m:
            command = m.group(1).strip()
            actions.append({"action": "compute", "command": command})
            i += 1
            continue

        # Check for PREDICT:
        if re.match(r'^PREDICT:', line, re.IGNORECASE):
            # Collect subsequent lines for coefficient parsing
            pred_lines = []
            i += 1
            while i < len(lines):
                pred_lines.append(lines[i])
                i += 1
            pred_text = "\n".join(pred_lines)
            prediction = _parse_prediction(pred_text)
            if prediction:
                actions.append({"action": "predict", **prediction})
            else:
                actions.append({
                    "action": "reasoning",
                    "text": f"PREDICT:\n{pred_text}  [could not parse coefficients]"
                })
            continue

        # Otherwise it's reasoning
        if line:
            actions.append({"action": "reasoning", "text": line})
        i += 1

    return actions


def _parse_prediction(text: str) -> Optional[dict]:
    """
    Extract coefficient arrays from prediction text.

    Looks for patterns like:
        a_coeffs = [1.0, 0.5, -0.3]
        a = [1.0, 0.5, -0.3]
    """
    result = {}
    for key in ["a", "b", "c", "f"]:
        # Try "X_coeffs = [...]" or "X = [...]"
        pattern = rf'{key}(?:_coeffs)?\s*=\s*\[([^\]]+)\]'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                vals = [float(v.strip()) for v in m.group(1).split(",")]
                result[key] = vals
            except ValueError:
                return None
        else:
            return None
    return result


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def format_query_result(response: dict) -> str:
    """Format a session query response as a message for the LLM."""
    if response["status"] == "ok":
        return (
            f"Query result for '{response['spec']}':\n"
            f"  integral_f_phi = {response['integral_f_phi']:+.12f}\n"
            f"  integral_u_phi = {response['integral_u_phi']:+.12f}\n"
            f"  (Query {response['query_number']}, "
            f"{response['queries_remaining']} remaining)"
        )
    else:
        return (
            f"Error: {response['message']}\n"
            f"  ({response['queries_remaining']} queries remaining, "
            f"errors don't count against budget)"
        )


def format_decompose_result(response: dict) -> str:
    """Format a DECOMPOSE response for the LLM."""
    if response["status"] == "ok":
        n_b = len(response["G_diff"])
        lines = [f"Decomposition for '{response['spec']}':"]
        lines.append(f"  {'j':<4} {'G_diff[j]':>16} {'G_adv[j]':>16} {'G_react[j]':>16} {'G_src[j]':>16}")
        lines.append("  " + "-" * 68)
        for j in range(n_b):
            lines.append(
                f"  {j:<4} {response['G_diff'][j]:>+16.10f} "
                f"{response['G_adv'][j]:>+16.10f} "
                f"{response['G_react'][j]:>+16.10f} "
                f"{response['G_src'][j]:>+16.10f}"
            )
        lines.append(f"\n  The equation for this test function is:")
        lines.append(f"    Σ_j a_j*G_diff[j] + Σ_j b_j*G_adv[j] + Σ_j c_j*G_react[j] = Σ_j f_j*G_src[j]")
        lines.append(f"  (Query {response['query_number']}, {response['queries_remaining']} remaining)")
        return "\n".join(lines)
    else:
        return (
            f"Error: {response['message']}\n"
            f"  ({response.get('queries_remaining', '?')} queries remaining, "
            f"errors don't count against budget)"
        )


def format_compute_result(response: dict) -> str:
    """Format a COMPUTE response for the LLM."""
    if response["status"] != "ok":
        return f"Compute error: {response['message']}"

    rtype = response.get("type", "")
    if rtype == "basis_info":
        lines = ["Basis information:"]
        for desc in response["descriptions"]:
            lines.append(f"  {desc}")
        lines.append(f"\nReference values:")
        lines.append("  " + response["reference_values"].replace("\n", "\n  "))
        return "\n".join(lines)
    elif rtype == "eval_basis":
        return f"Basis function values:\n  {response['table'].replace(chr(10), chr(10) + '  ')}"
    elif rtype == "eval_solution":
        return f"Solution values:\n  {response['table'].replace(chr(10), chr(10) + '  ')}"
    elif rtype == "solve":
        lines = [
            f"Solve result ({response['n_rows']} equations, {response['n_unknowns']} unknowns):",
            f"  a_coeffs = {response['a_coeffs']}",
            f"  b_coeffs = {response['b_coeffs']}",
            f"  c_coeffs = {response['c_coeffs']}",
            f"  f_coeffs = {response['f_coeffs']}",
        ]
        return "\n".join(lines)
    else:
        return str(response)


def format_score_result(result: dict) -> str:
    """Format the final scoring result."""
    return (
        f"=== FINAL SCORE ===\n\n"
        f"{result['summary']}\n\n"
        f"Ground truth PDE:\n{result['pde_description']}\n\n"
        f"Queries used: {result['queries_used']}"
    )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """Calls the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install the anthropic SDK: pip install anthropic\n"
                "And set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class MockBackend:
    """
    A scripted mock LLM for testing the loop without API access.

    Demonstrates the recommended strategy:
        1. COMPUTE basis_info to understand the basis
        2. DECOMPOSE with well-chosen test functions to build the linear system
        3. Submit prediction based on the collected data
    """

    def __init__(self):
        self.call_count = 0
        self.script = [
            # Turn 1: get basis info (free, no query cost)
            "Let me start by understanding the basis functions.\n\n"
            "COMPUTE: basis_info",
            # Turns 2-13: one DECOMPOSE per turn (12 equations for 12 unknowns at medium/n_basis=3)
            "Good. Now I'll build the linear system one row at a time.\n\n"
            "DECOMPOSE: sin(pi*x)",
            "DECOMPOSE: sin(2*pi*x)",
            "DECOMPOSE: sin(3*pi*x)",
            "DECOMPOSE: x*(1-x)",
            "DECOMPOSE: x*(1-x)*(2*x-1)",
            "DECOMPOSE: x**2*(1-x)**2",
            "DECOMPOSE: x*(1-x)*sin(pi*x)",
            "DECOMPOSE: x*(1-x)*exp(-20*(x-0.3)**2)",
            "DECOMPOSE: x*(1-x)*sin(2*pi*x)",
            "DECOMPOSE: sin(4*pi*x)",
            "DECOMPOSE: x*(1-x)*(6*x**2-6*x+1)",
            "DECOMPOSE: x**2*(1-x)**2*sin(pi*x)",
            # Turn 14: ask the server to solve, get back a PREDICT block
            "I have collected 12 equations for 12 unknowns. Let me ask the server to solve.\n\n"
            "COMPUTE: solve",
            # Turn 15: PREDICT alone — copy-paste from the solve output (dummy values for mock)
            "The server solved the system. Submitting the suggested coefficients.\n\n"
            "PREDICT:\n"
            "a_coeffs = [0.5, 0.0, 0.1]\n"
            "b_coeffs = [-1.0, 0.5, -1.0]\n"
            "c_coeffs = [0.0, -0.5, -0.5]\n"
            "f_coeffs = [-0.6, -0.8, -1.2]",
        ]

    def chat(self, system: str, messages: list[dict]) -> str:
        idx = min(self.call_count, len(self.script) - 1)
        self.call_count += 1
        return self.script[idx]


# ---------------------------------------------------------------------------
# Post-hoc efficiency analysis
# ---------------------------------------------------------------------------

def compute_efficiency_curve(session) -> dict:
    """
    Post-hoc analysis: what was the coefficient error after each query?
    Uses stored G matrices from both DECOMPOSE and QUERY history.
    The model never sees this — it's for benchmark scoring only.
    """
    import numpy as np

    basis = session.basis
    n_b = basis.n_basis
    pde = session.pde

    # Collect rows from DECOMPOSE history and QUERY history (both store G matrices)
    history_rows = []
    for r in session.history:
        if r.status != "ok":
            continue
        resp = r.response
        if resp.get("type") == "decompose":
            history_rows.append({
                "G_diff": resp["G_diff"],
                "G_adv":  resp["G_adv"],
                "G_react":resp["G_react"],
                "G_src":  resp["G_src"],
                "integral_f_phi": resp["integral_f_phi"],
            })
        elif "integral_f_phi" in resp and "_G_diff" in resp:
            history_rows.append({
                "G_diff": resp["_G_diff"],
                "G_adv":  resp["_G_adv"],
                "G_react":resp["_G_react"],
                "G_src":  resp["_G_src"],
                "integral_f_phi": resp["integral_f_phi"],
            })

    if not history_rows:
        return {"queries": [], "errors": [], "auc": float('inf')}

    def get_matrices(rows):
        G_diff  = np.array([r["G_diff"]  for r in rows])
        G_adv   = np.array([r["G_adv"]   for r in rows])
        G_react = np.array([r["G_react"] for r in rows])
        G_src   = np.array([r["G_src"]   for r in rows])
        rhs     = np.array([r["integral_f_phi"] for r in rows])
        return G_diff, G_adv, G_react, G_src, rhs

    true_coeffs = np.concatenate([
        pde.a.coeffs, pde.b.coeffs, pde.c.coeffs, pde.f.coeffs
    ])

    errors = []
    queries = []

    for k in range(1, len(history_rows) + 1):
        if k < n_b:
            errors.append(float('inf'))
            queries.append(k)
            continue

        try:
            G_diff, G_adv, G_react, G_src, rhs = get_matrices(history_rows[:k])
            f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs, rcond=None)
            A_lhs = np.hstack([G_diff, G_adv, G_react])
            theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs, rcond=None)
            recovered = np.concatenate([theta, f_rec])
            err = float(np.max(np.abs(recovered - true_coeffs)))
        except Exception:
            err = float('inf')

        errors.append(err)
        queries.append(k)

    finite_errors = [e for e in errors if np.isfinite(e)]
    auc = float(np.trapezoid(finite_errors)) / max(len(finite_errors), 1) if finite_errors else float('inf')

    return {
        "queries": queries,
        "errors": errors,
        "auc": auc,
        "final_error": errors[-1] if errors else float('inf'),
        "queries_to_below_1":   next((q for q, e in zip(queries, errors) if e < 1.0),  None),
        "queries_to_below_0.1": next((q for q, e in zip(queries, errors) if e < 0.1), None),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_session(
    session: InteractiveSession,
    backend,
    verbose: bool = True,
    max_turns: int = 50,
    hints: bool = False,
    no_decompose: bool = False,
) -> dict:
    """
    Run the interactive LLM session.

    Returns a dict with the session transcript, score, and metadata.
    """
    system_prompt = session.system_prompt()

    # Apply no_decompose flag to session before generating system prompt
    session.decompose_enabled = not no_decompose

    # Add protocol instructions to system prompt
    if no_decompose:
        protocol = """

## Communication protocol

To probe the PDE with a test function, write:
    QUERY: <math expression>

To request free calculations, write:
    COMPUTE: basis_info
    COMPUTE: eval_basis 0.1 0.25 0.5 0.75 0.9
    COMPUTE: eval_solution 0.1 0.25 0.5 0.75 0.9
    COMPUTE: solve

To submit your final prediction, write:
    PREDICT:
    a_coeffs = [c0, c1, ...]
    b_coeffs = [c0, c1, ...]
    c_coeffs = [c0, c1, ...]
    f_coeffs = [c0, c1, ...]

Rules:
- ONE QUERY per response.
- You may NOT PREDICT in the same response as a QUERY.
- COMPUTE is free and unlimited.
"""
    else:
        protocol = """

## Communication protocol

To probe the PDE with a test function, write:
    DECOMPOSE: <math expression>

To request free calculations, write:
    COMPUTE: basis_info
    COMPUTE: eval_basis 0.1 0.25 0.5 0.75 0.9
    COMPUTE: eval_solution 0.1 0.25 0.5 0.75 0.9
    COMPUTE: solve

To submit a simple query (returns only ∫fφ and ∫uφ), write:
    QUERY: <math expression>

To submit your final prediction, write:
    PREDICT:
    a_coeffs = [c0, c1, ...]
    b_coeffs = [c0, c1, ...]
    c_coeffs = [c0, c1, ...]
    f_coeffs = [c0, c1, ...]

Rules:
- ONE QUERY or DECOMPOSE per response.
- You may NOT PREDICT in the same response as a QUERY or DECOMPOSE.
- COMPUTE is free and unlimited.
"""

    if hints:
        protocol += """
COMPUTE: solve is free and can be called at any time. You should call it after
collecting enough equations to check your progress. If the condition number is
high, add more equations before submitting.
"""

    full_system = system_prompt + protocol

    messages = [{"role": "user", "content": "Begin. You have a limited query budget. Your score depends on both accuracy and efficiency — every query should be maximally informative. Choose your test functions carefully."}]
    turn = 0
    done = False

    if verbose:
        print("=" * 70)
        print("LLM SESSION START")
        print("=" * 70)
        print(f"Difficulty: {session.difficulty}, "
              f"n_basis: {session.basis.n_basis}, "
              f"budget: {session.max_queries}")
        print()

    last_solve_coeffs = [None]  # mutable container to track across turns

    while not done and turn < max_turns:
        turn += 1

        # Get LLM response
        if verbose:
            print(f"--- Turn {turn} ---")

        llm_text = backend.chat(full_system, messages)

        if verbose:
            # Print LLM response (truncated)
            display = llm_text[:500] + ("..." if len(llm_text) > 500 else "")
            print(f"LLM: {display}\n")

        # Add LLM response to conversation
        messages.append({"role": "assistant", "content": llm_text})

        # Parse actions
        actions = parse_llm_response(llm_text)

        # Execute actions and collect results
        result_parts = []
        query_executed = False  # tracks whether a QUERY/DECOMPOSE ran this turn

        # Pass 1: execute all COMPUTE actions (free, unlimited)
        for act in actions:
            if act["action"] == "compute":
                resp = session.compute(act["command"])
                result_text = format_compute_result(resp)
                result_parts.append(result_text)
                if verbose:
                    print(f"  → {result_text}\n")

        # Pass 2: first QUERY or DECOMPOSE only; then PREDICT (only if no query this turn)
        for act in actions:
            if act["action"] == "compute":
                continue  # already handled above

            elif act["action"] == "decompose" and no_decompose:
                msg = "DECOMPOSE is not available. Use QUERY to probe the PDE, then COMPUTE: solve to recover coefficients."
                result_parts.append(msg)
                if verbose:
                    print(f"  [blocked] {msg}\n")

            elif act["action"] in ("query", "decompose"):
                if not query_executed:
                    if act["action"] == "query":
                        resp = session.query(act["spec"])
                        result_text = format_query_result(resp)
                    else:
                        resp = session.decompose(act["spec"])
                        result_text = format_decompose_result(resp)
                    result_parts.append(result_text)
                    query_executed = True
                    if verbose:
                        print(f"  → {result_text}\n")

                    # Auto-feedback: show solve progress periodically
                    if resp.get("status") == "ok":
                        n_probes = sum(
                            1 for r in session.history
                            if r.status == "ok" and (
                                r.response.get("type") == "decompose"
                                or "integral_f_phi" in r.response
                            )
                        )
                        feedback_interval = session.basis.n_basis
                        if n_probes >= session.basis.n_basis and n_probes % feedback_interval == 0:
                            solve_resp = session.compute("solve")
                            if solve_resp.get("status") == "ok":
                                solve_text = format_compute_result(solve_resp)

                                # Diff against previous solve
                                current_coeffs = {
                                    "a": solve_resp["a_coeffs"],
                                    "b": solve_resp["b_coeffs"],
                                    "c": solve_resp["c_coeffs"],
                                    "f": solve_resp["f_coeffs"],
                                }
                                if last_solve_coeffs[0] is not None:
                                    diff_lines = ["Coefficient changes since last solve:"]
                                    for name in ["a", "b", "c", "f"]:
                                        changes = [
                                            abs(c - p) for c, p
                                            in zip(current_coeffs[name], last_solve_coeffs[0][name])
                                        ]
                                        max_change = max(changes)
                                        diff_lines.append(
                                            f"  {name}_coeffs max change: {max_change:.6f}"
                                        )
                                    solve_text += "\n" + "\n".join(diff_lines)

                                last_solve_coeffs[0] = current_coeffs

                                result_parts.append(
                                    f"\n--- Auto-progress (after {n_probes} probes) ---\n"
                                    + solve_text
                                )
                                if verbose:
                                    print(f"  [auto-progress] {n_probes} probes collected\n")
                                    print(f"  → {solve_text}\n")

                else:
                    msg = (
                        "One QUERY/DECOMPOSE per turn. "
                        "Review the result above, then submit your next action."
                    )
                    result_parts.append(msg)
                    if verbose:
                        print(f"  [skipped extra {act['action'].upper()}] {msg}\n")

            elif act["action"] == "predict":
                if query_executed:
                    msg = (
                        "You submitted a PREDICT in the same turn as a DECOMPOSE/QUERY. "
                        "Review your results first, then PREDICT in a separate response."
                    )
                    result_parts.append(msg)
                    if verbose:
                        print(f"  [skipped PREDICT] {msg}\n")
                else:
                    prediction = {k: act[k] for k in ["a", "b", "c", "f"]}
                    result = session.submit_prediction(prediction)
                    if result["status"] == "scored":
                        score_text = format_score_result(result)
                        result_parts.append(score_text)
                        done = True
                        if verbose:
                            print(f"\n{score_text}\n")
                    else:
                        result_parts.append(f"Error: {result['message']}")

        # If no actions found, nudge the LLM
        if not any(a["action"] in ("query", "decompose", "compute", "predict") for a in actions):
            result_parts.append(
                "I didn't see a QUERY, DECOMPOSE, COMPUTE, or PREDICT in your response. "
                "Please submit a test function with 'QUERY: <expression>' or "
                "'DECOMPOSE: <expression>', use 'COMPUTE: basis_info', "
                "or submit your final answer with 'PREDICT:'."
            )

        # Add results to conversation
        if result_parts and not done:
            user_msg = "\n\n".join(result_parts)
            messages.append({"role": "user", "content": user_msg})

        # Check if budget exhausted
        if session.queries_used >= session.max_queries and not session.prediction_submitted:
            messages.append({
                "role": "user",
                "content": (
                    f"Your query budget is exhausted ({session.max_queries} queries used). "
                    f"Please submit your best prediction now using PREDICT:."
                ),
            })

    # Build result
    output = {
        "turns": turn,
        "queries_used": session.queries_used,
        "prediction_submitted": session.prediction_submitted,
        "transcript": session.to_transcript(),
        "messages": messages,
    }
    if session.final_score:
        output["score"] = session.final_score.to_dict()

    # Store the last submitted prediction for downstream use (e.g. plotting)
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            for act in reversed(parse_llm_response(msg["content"])):
                if act["action"] == "predict":
                    output["prediction"] = {k: act[k] for k in ["a", "b", "c", "f"]}
                    break
        if "prediction" in output:
            break

    # Post-hoc efficiency analysis
    output["efficiency"] = compute_efficiency_curve(session)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LLM PDE identification session")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API needed)")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=30)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--save-transcript", type=str, default=None,
                        help="Save session transcript to file")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save results plot to file (e.g. results.png)")
    parser.add_argument("--hints", action="store_true",
                        help="Add strategy hints to the system prompt (Round 1/2/3 workflow)")
    parser.add_argument("--no-decompose", action="store_true",
                        help="Disable DECOMPOSE; COMPUTE: solve builds system from QUERY history instead")
    args = parser.parse_args()

    # Create session
    session = InteractiveSession.from_difficulty(
        difficulty=args.difficulty,
        seed=args.seed,
        max_queries=args.max_queries,
    )

    # Create backend
    if args.mock:
        backend = MockBackend()
    else:
        backend = AnthropicBackend(model=args.model)

    # Run
    result = run_session(
        session=session,
        backend=backend,
        verbose=True,
        max_turns=args.max_turns,
        hints=args.hints,
        no_decompose=args.no_decompose,
    )

    # Save transcript
    if args.save_transcript:
        with open(args.save_transcript, "w") as f:
            f.write(result["transcript"])
        print(f"\nTranscript saved to {args.save_transcript}")

    # Generate results plot
    if session.prediction_submitted and "prediction" in result:
        plot_path = args.save_plot
        if plot_path is None:
            plot_path = f"results_{args.difficulty}_{args.seed}.png"

        try:
            import matplotlib
            matplotlib.use("Agg")
            from diagnostics import plot_session_results
            fig = plot_session_results(session, result["prediction"], save_path=plot_path)
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            print(f"Results plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Session complete: {result['turns']} turns, "
          f"{result['queries_used']} queries used")
    if "score" in result:
        print(f"Total coefficient error: {result['score']['total_coeff_error']:.4f}")
        print(f"Total pointwise error:   {result['score']['total_pointwise_error']:.4f}")
    if "efficiency" in result:
        eff = result["efficiency"]
        print(f"Efficiency AUC:          {eff['auc']:.4f}")
        print(f"Queries to error < 1.0:  {eff['queries_to_below_1']}")
        print(f"Queries to error < 0.1:  {eff['queries_to_below_0.1']}")


if __name__ == "__main__":
    main()