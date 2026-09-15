"""Does a NONLINEAR length correction do what the scalar one cannot?

Section 4 runs the scalar length-control recipe head to head against the multivariate
phi and finds the scalar one leaves every format effect significant, moving the
docstring and bare-comment effects in opposite directions. The paper attributes that to
the response being non-monotone in added text, and to comment style rather than length
being the discriminating variable. The natural objection is that a nonlinear length
decoupling would handle a non-monotone response, and the paper only cites that line
without running it.

This runs it, on the same design, same cached scores, same paired-effect machinery as
length_only_baseline.py. Four corrections, each cross-fitted on the same folds:

  linear      ridge on character count            (the scalar recipe, for reference)
  isotonic    monotone but nonlinear in length
  spline      natural cubic basis on length + ridge
  boosted     gradient boosting on length alone   (the flexible ceiling)

The last is the strongest correction a length-only covariate admits. If it flattens the
four format effects, the paper's claim narrows to "linear is not enough". If it does
not, the claim that the discriminating variable is style and not length is measured
rather than asserted.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "nonlinear_length_baseline.json")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

sys.path.insert(0, SCRIPTS)
from rm_code_build import run_tests, body_indent  # noqa: E402
from rm_code_extra import format_features  # noqa: E402


def light_verbosify(code):
    lines = code.split("\n")
    out, added = [], False
    for i, ln in enumerate(lines):
        out.append(ln)
        st = ln.strip()
        if not added and st.startswith("def ") and st.endswith(":"):
            out.append(body_indent(lines, i) + "# main computation below")
            added = True
    return "\n".join(out)


rows = json.load(open(os.path.join(CACHE, "mbpp_rows.json")))
from datasets import load_dataset  # noqa: E402
d = load_dataset("google-research-datasets/mbpp", "full", split="test")
tests_by_prompt = {r["text"]: (r.get("test_setup_code", ""), r["test_list"]) for r in d}

light_rows, by_q = [], {}
for r in rows:
    by_q.setdefault(r[3], []).append(r)
for q, group in by_q.items():
    cells = {r[4]: r for r in group}
    prompt = group[0][0]
    setup, tests = tests_by_prompt[prompt]
    lc = light_verbosify(cells["correct-terse"][1])
    lb = light_verbosify(cells["buggy-terse"][1])
    if run_tests(lc, setup, tests) and not run_tests(lb, setup, tests):
        light_rows.append([prompt, lc, 1, q, "correct-light"])
        light_rows.append([prompt, lb, 0, q, "buggy-light"])
print(f"light problems kept: {len(light_rows) // 2}", flush=True)

all_rows = rows + light_rows
code = [r[1] for r in all_rows]
qid = np.array([int(r[3]) for r in all_rows])
cell = np.array([r[4] for r in all_rows])
s = np.concatenate([np.load(os.path.join(CACHE, "mbpp_rm_s.npy")),
                    np.load(os.path.join(CACHE, "mbpp_rm_light_s.npy"))])
assert len(s) == len(all_rows)

phi_full = np.array([format_features(c) for c in code], float)
L = np.array([[len(c)] for c in code], float)


def iso_cv(x, y):
    """Isotonic has no multi-output CV path in cross_val_predict; fold it by hand."""
    p = np.zeros_like(y)
    for tr, te in CV.split(x):
        m = IsotonicRegression(out_of_bounds="clip").fit(x[tr, 0], y[tr])
        p[te] = m.predict(x[te, 0])
    return p


scores = {
    "raw": s,
    "resid_full_phi": s - cross_val_predict(Ridge(alpha=1.0), phi_full, s, cv=CV),
    "resid_length_linear": s - cross_val_predict(Ridge(alpha=1.0), L, s, cv=CV),
    "resid_length_isotonic": s - iso_cv(L, s),
    "resid_length_spline": s - cross_val_predict(
        make_pipeline(SplineTransformer(n_knots=8, degree=3), Ridge(alpha=1.0)), L, s, cv=CV),
    "resid_length_boosted": s - cross_val_predict(
        GradientBoostingRegressor(random_state=0), L, s, cv=CV),
}

# how much of the score each length model explains, for context
r2 = {}
for k, u in scores.items():
    if k == "raw":
        continue
    g = s - u
    r2[k] = round(float(1 - ((s - g) ** 2).sum() / ((s - s.mean()) ** 2).sum()), 4)


def paired(u, hi_cell, lo_cell, B=2000):
    per = []
    for t in np.unique(qid):
        a = u[(qid == t) & (cell == hi_cell)]
        b = u[(qid == t) & (cell == lo_cell)]
        if len(a) and len(b):
            per.append(a.mean() - b.mean())
    per = np.array(per)
    bo = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)]
    return {"mean": round(float(per.mean()), 4),
            "CI": [round(float(np.percentile(bo, 2.5)), 4),
                   round(float(np.percentile(bo, 97.5)), 4)]}


res = {"R2_of_correction": r2, "effects": {}}
for kind, u in scores.items():
    for lab in ("correct", "buggy"):
        for dose, hi in (("light", f"{lab}-light"), ("heavy", f"{lab}-verbose")):
            res["effects"][f"{lab}_{dose}_{kind}"] = paired(u, hi, f"{lab}-terse")
    res["effects"][f"margin_terse_{kind}"] = paired(u, "correct-terse", "buggy-terse")
    res["effects"][f"margin_verbose_{kind}"] = paired(u, "correct-verbose", "buggy-verbose")

# the headline: how many of the four format effects each correction flattens to zero
summary = {}
for kind in scores:
    flat = 0
    for lab in ("correct", "buggy"):
        for dose in ("light", "heavy"):
            ci = res["effects"][f"{lab}_{dose}_{kind}"]["CI"]
            if ci[0] <= 0 <= ci[1]:
                flat += 1
    summary[kind] = {"format_effects_covering_zero": flat, "of": 4}
res["summary"] = summary

json.dump(res, open(OUT, "w"), indent=1)
print("\nR2 of each correction:", json.dumps(r2, indent=1))
print("\nformat effects covering zero (of four):")
for k, v in summary.items():
    print("  %-24s %d/4" % (k, v["format_effects_covering_zero"]))
print("\nwrote", OUT)
