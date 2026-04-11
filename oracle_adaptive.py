"""oracle_adaptive.py — Test strategies that use solution shape."""
import numpy as np
from main_loop import ProbeSession

for seed in [1, 4, 9]:  # seed 4 is the hard one
    session = ProbeSession.from_difficulty("extreme", seed, 48)
    u_vals = session.solution.u
    x_vals = session.solution.x
    
    # Find where |u| is large and small
    u_abs = np.abs(u_vals)
    peaks = x_vals[np.argsort(u_abs)[-5:]]  # 5 points where |u| biggest
    zeros = x_vals[(u_abs < np.percentile(u_abs, 20)) & (x_vals > 0.05) & (x_vals < 0.95)]
    
    print(f"\nseed {seed}: |u| peaks at x={np.sort(peaks)}, near-zero at x={zeros[:5]}")
    
    # Strategy 5: localized at u-peaks (maximize reaction signal)
    s5 = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 25):
        s5.query(f"sin({k}*pi*x)")
    for n in range(1, 9):
        s5.query(f"x**{n}*(1-x)")
    # Localized at peaks of |u|
    for c in peaks[:6]:
        s5.query(f"exp(-50*(x-{c:.3f})**2)*x*(1-x)")
    # Fill remaining with exponentials
    remaining = 48 - s5.queries_used
    for a in range(1, remaining + 1):
        s5.query(f"exp({a}*x)*x*(1-x)")
    r5 = s5.solve()
    
    # Strategy 6: localized at u-zeros (isolate a,b from c)
    s6 = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 25):
        s6.query(f"sin({k}*pi*x)")
    for n in range(1, 9):
        s6.query(f"x**{n}*(1-x)")
    for c in zeros[:6]:
        s6.query(f"exp(-50*(x-{c:.3f})**2)*x*(1-x)")
    remaining = 48 - s6.queries_used
    for a in range(1, remaining + 1):
        s6.query(f"exp({a}*x)*x*(1-x)")
    r6 = s6.solve()
    
    # Compare to previous strategies
    s_harm = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 49):
        s_harm.query(f"sin({k}*pi*x)")
    r_harm = s_harm.solve()
    
    s_div = ProbeSession.from_difficulty("extreme", seed, 48)
    for k in range(1, 25):
        s_div.query(f"sin({k}*pi*x)")
    for n in range(1, 9):
        s_div.query(f"x**{n}*(1-x)")
    for c in [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]:
        s_div.query(f"exp(-50*(x-{c})**2)*x*(1-x)")
    for a in [1, 2, -1, -2]:
        s_div.query(f"exp({a}*x)*x*(1-x)")
    r_div = s_div.solve()
    
    def score(session, solve_result):
        if solve_result.get("status") != "ok":
            return float("inf")
        pred = {
            "a": solve_result["a_coeffs"],
            "b": solve_result["b_coeffs"],
            "c": solve_result["c_coeffs"],
            "f": solve_result["f_coeffs"],
        }
        return session.submit_prediction(pred)["score"]["total_coeff_error"]

    print(f"  harmonics:       {score(s_harm, r_harm):.6f}")
    print(f"  diverse (blind): {score(s_div, r_div):.6f}")
    print(f"  peak-localized:  {score(s5, r5):.6f}")
    print(f"  zero-localized:  {score(s6, r6):.6f}")