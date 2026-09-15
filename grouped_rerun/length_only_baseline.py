"""Internal review round 6: scalar length-only correction vs multivariate phi,
head-to-head on the reward-model edit-type design.

Length-controlled evaluation (LC-AlpacaEval style) corrects a judge score on
a single scalar length covariate. The edit-type experiment (rm_code_dose.py)
found the base RM's format response is non-monotone in added text (a bare
comment line is penalized, docstring-style verbosity rewarded), which a
monotone-in-length correction should be unable to remove. This measures that
directly: the four paired (label x dose) format effects under (a) raw score,
(b) residualization on the full nine-feature phi, (c) residualization on
scalar length alone (character count; token count reported as a check).

Same candidates, cached scores, and paired-effect machinery as
rm_code_dose.py.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "length_only_baseline.json")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

sys.path.insert(0, SCRIPTS)
from rm_code_build import run_tests, body_indent  # noqa: E402
from rm_code_extra import format_features  # noqa: E402


def light_verbosify(code):
    lines = code.split("\n")
    out = []
    added = False
    for i, ln in enumerate(lines):
        out.append(ln)
        st = ln.strip()
        if not added and st.startswith("def ") and st.endswith(":"):
            out.append(body_indent(lines, i) + "# main computation below")
            added = True
    return "\n".join(out)


rows = json.load(open(os.path.join(CACHE, "mbpp_rows.json")))
from datasets import load_dataset
d = load_dataset("google-research-datasets/mbpp", "full", split="test")
tests_by_prompt = {r["text"]: (r.get("test_setup_code", ""), r["test_list"])
                   for r in d}

light_rows = []
by_q = {}
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
phi_chars = np.array([[len(c)] for c in code], float)
phi_tokens = np.array([[len(c.split())] for c in code], float)

scores = {
    "raw": s,
    "resid_full_phi": s - cross_val_predict(Ridge(alpha=1.0), phi_full, s, cv=CV),
    "resid_length_chars": s - cross_val_predict(Ridge(alpha=1.0), phi_chars, s, cv=CV),
    "resid_length_tokens": s - cross_val_predict(Ridge(alpha=1.0), phi_tokens, s, cv=CV),
}


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


res = {}
for kind, u in scores.items():
    for lab in ("correct", "buggy"):
        for dose, hi in (("light", f"{lab}-light"), ("heavy", f"{lab}-verbose")):
            res[f"{lab}_{dose}_{kind}"] = paired(u, hi, f"{lab}-terse")
    # construct margins as the preservation check
    res[f"margin_terse_{kind}"] = paired(u, "correct-terse", "buggy-terse")
    res[f"margin_verbose_{kind}"] = paired(u, "correct-verbose", "buggy-verbose")

json.dump(res, open(OUT, "w"), indent=1)
for k, v in res.items():
    print(k, v)
print("wrote", OUT)
