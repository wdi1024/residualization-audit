"""Verbosity dose-response for the reward-model code audit.

The main 2x2 design manipulates format at two doses: none (terse) and heavy
(docstring + comments + trailing comment). This adds an intermediate dose---
a SINGLE inline comment line---to both the correct and the buggy member of
every MBPP problem, labels re-verified by unit-test execution as always.

Prediction under the paper's reading: the raw format effect is monotone in
dose (0 < light < heavy) because the reward model rides the format channel;
the residualized effect is flat at zero across doses. The residualizer is
the unchanged cross-fitted ridge, refit on the expanded candidate set
(six cells per problem).

Writes ../notes/rm_code_dose.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "notes", "rm_code_dose.json")
SCACHE = os.path.join(CACHE, "mbpp_rm_light_s.npy")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

sys.path.insert(0, HERE)
from rm_code_build import run_tests, body_indent  # noqa: E402
from rm_code_extra import format_features  # noqa: E402


def light_verbosify(code):
    """Single comment line after the first def: the intermediate dose."""
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

# build light variants from the terse members, re-verify labels
light_rows = []
kept = dropped = 0
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
        kept += 1
    else:
        dropped += 1
print(f"light variants: kept={kept} problems, dropped={dropped}")

all_rows = rows + light_rows
prompt = [r[0] for r in all_rows]
code = [r[1] for r in all_rows]
qid = np.array([int(r[3]) for r in all_rows])
cell = np.array([r[4] for r in all_rows])

# score: reuse cached base-RM scores for the original rows, score only light
s_orig = np.load(os.path.join(CACHE, "mbpp_rm_s.npy"))
if os.path.exists(SCACHE):
    s_light = np.load(SCACHE)
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "OpenAssistant/reward-model-deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    lp = [r[0] for r in light_rows]
    lc_ = [r[1] for r in light_rows]
    s_light = np.zeros(len(light_rows))
    with torch.no_grad():
        for i in range(0, len(light_rows), 16):
            enc = tok(lp[i:i + 16], lc_[i:i + 16], return_tensors="pt",
                      truncation=True, max_length=512, padding=True)
            s_light[i:i + 16] = model(**enc).logits[:, 0].numpy()
            if i % 320 == 0:
                print(f"scored {i}/{len(light_rows)}", flush=True)
    np.save(SCACHE, s_light)
s = np.concatenate([s_orig, s_light])

phi = np.array([format_features(c) for c in code], float)
srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)


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
                   round(float(np.percentile(bo, 97.5)), 4)],
            "n_problems": int(len(per))}


res = {"n_light_problems": kept}
for lab in ("correct", "buggy"):
    for dose, hi in (("light", f"{lab}-light"), ("heavy", f"{lab}-verbose")):
        for u, kind in ((s, "raw"), (srep, "residualized")):
            res[f"{lab}_{dose}_{kind}"] = paired(u, hi, f"{lab}-terse")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
