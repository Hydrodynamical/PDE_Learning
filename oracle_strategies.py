"""oracle_strategies.py — Compare fixed query strategies across seeds."""
import numpy as np
from main_loop import ProbeSession

results = []
for seed in range(1, 11):
    # Strategy 1: pure harmonics (Flash's playbook)
    s1 = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 49):
        s1.query(f"sin({k}*pi*x)")
    r1 = s1.solve()

    # Strategy 2: pure polynomials
    s2 = ProbeSession.from_difficulty("extreme", seed, 48)
    for n in range(1, 25):
        s2.query(f"x**{n}*(1-x)")
    for n in range(1, 25):
        s2.query(f"x*(1-x)**{n}")
    r2 = s2.solve()

    # Strategy 3: mixed (32 harmonics + 16 polynomials)
    s3 = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 33):
        s3.query(f"sin({k}*pi*x)")
    for n in range(1, 9):
        s3.query(f"x**{n}*(1-x)")
    for n in range(1, 9):
        s3.query(f"x*(1-x)**{n}")
    r3 = s3.solve()

    # Strategy 4: diverse (harmonics + polynomials + localized + exponential)
    s4 = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 25):
        s4.query(f"sin({k}*pi*x)")
    for n in range(1, 9):
        s4.query(f"x**{n}*(1-x)")
    for c in [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]:
        s4.query(f"exp(-50*(x-{c})**2)*x*(1-x)")
    for a in [1, 2, -1, -2]:
        s4.query(f"exp({a}*x)*x*(1-x)")
    r4 = s4.solve()

    def score(session, solve_result):
        if solve_result.get("status") != "ok":
            return float("inf")
        pred = {
            "a": solve_result["a_coeffs"],
            "b": solve_result["b_coeffs"],
            "c": solve_result["c_coeffs"],
            "f": solve_result["f_coeffs"],
        }
        result = session.submit_prediction(pred)
        return result["score"]["total_coeff_error"]

    e1 = score(s1, r1)
    e2 = score(s2, r2)
    e3 = score(s3, r3)
    e4 = score(s4, r4)

    results.append((seed, e1, e2, e3, e4))
    print(f"  seed {seed:>2}: harmonics={e1:.6f}  polys={e2:.6f}  mixed={e3:.6f}  diverse={e4:.6f}")

print(f"\n{'Strategy':<15s}  {'Median':>10s}  {'Mean':>10s}  {'Max':>10s}")
print(f"{'─'*15}  {'─'*10}  {'─'*10}  {'─'*10}")
for name, idx in [("Harmonics", 1), ("Polynomials", 2), ("Mixed", 3), ("Diverse", 4)]:
    vals = [r[idx] for r in results]
    print(f"{name:<15s}  {np.median(vals):>10.6f}  {np.mean(vals):>10.6f}  {max(vals):>10.6f}")