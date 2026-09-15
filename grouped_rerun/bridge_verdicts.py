"""Committed-procedure verdict for every cell of the assignment sweep
(paper v165, Section 6.1). Reads bridge_scorer_sweep.json only; no refit.
Gates: router A(a-hat) >= 0.65; screen R2 >= 0.05 AND Delta_slice
(= A_full_raw - A_slice_raw) >= 0.03.

Output (2026-09-07 run): no cell reaches the checks. Router refuses all
scorers at p <= 0.6 (A(a-hat) <= 0.565); at p >= 0.7 the slice-degradation
statistic refuses every cell (max +0.028, reward model p=0.8; for the
judges it is mostly negative)."""
import json, os
d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "bridge_scorer_sweep.json")))
for sc in ["rm", "haiku45", "sonnet5", "opus5"]:
    for r in d[sc]:
        if r["n_slice"] == 0:
            print(f"{sc:8s} p={r['p']}: slice empty -> screened out (degenerate)"); continue
        Aa, R2 = r["C1"], r["R2_phi_s"]
        Ds = r["A_full_raw"] - r["A_slice_raw"]
        if Aa < 0.65: v = "screened out (router)"
        elif R2 < 0.05: v = "not validated (screen R2)"
        elif Ds < 0.03: v = "not validated (screen Delta_slice)"
        else: v = "-> post-adjustment checks"
        print(f"{sc:8s} p={r['p']}: A(a)={Aa:.3f} R2={R2:.3f} Dslice={Ds:+.3f} "
              f"gain_slice={r['gain_slice']:+.3f} gain_full={r['gain_full']:+.3f}  {v}")
