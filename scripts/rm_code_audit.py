"""Card C, stage 2: audit a public reward model on the MBPP 2x2 setting.

Setting: candidates from rm_code_build.py -- per problem a 2x2 cell design
{correct, buggy} x {terse, verbose}, with correctness verified by unit-test
execution (a mechanical y independent of any text overlap with the prompt)
and format varied by semantics-preserving comment/docstring edits.

Score s: a public preference reward model (OpenAssistant deberta-v3 RM),
scoring (problem statement, code) pairs. phi: format-only surface features
(lengths, comment density, docstring presence, blank lines) -- construct-
external for correctness by construction: within every problem, format is
manipulated at fixed semantics.

Protocol (frozen recipe): the C1 screen is computed and the prediction
committed BEFORE the reward model produces a single score, exactly as in the
held-out QA replications. Because format is orthogonal to correctness by
design, C1 is expected to sit at chance -- the real-data instantiation of
the rho~0 regime, where the label-proxy screen is blind to a channel the
scorer may still ride. The audit then measures:
  - R2(phi -> s): the RM's format-loading
  - paired format effect (verbose - terse at fixed correctness) and paired
    construct effect (correct - buggy at fixed format), raw vs residualized,
    with problem-level bootstrap CIs -- a causal decomposition the 2x2
    design supports directly
  - full-population alignment A(y, s) vs A(y, s_rep), problem-clustered CI
  - pairwise preference accuracy P(s_correct > s_buggy) within cells

Writes ../notes/rm_code_audit.json (+ .committed).
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "notes", "rm_code_audit.json")
SCACHE = os.path.join(CACHE, "mbpp_rm_s.npy")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

rows = json.load(open(os.path.join(CACHE, "mbpp_rows.json")))
prompt = [r[0] for r in rows]
code = [r[1] for r in rows]
y = np.array([int(r[2]) for r in rows])
qid = np.array([int(r[3]) for r in rows])
cell = np.array([r[4] for r in rows])
n = len(rows)
print(f"n={n} candidates, {len(np.unique(qid))} problems")


def format_features(c):
    lines = c.split("\n")
    nonblank = [l for l in lines if l.strip()]
    comments = [l for l in lines if l.strip().startswith("#")]
    in_doc = '"""' in c
    ncomm_chars = sum(len(l) for l in comments)
    return [len(c), len(lines), len(c.split()),
            len(comments), ncomm_chars / max(1, len(c)),
            1.0 if in_doc else 0.0,
            len(lines) - len(nonblank),
            np.mean([len(l) for l in nonblank]) if nonblank else 0.0,
            max((len(l) for l in nonblank), default=0)]


phi = np.array([format_features(c) for c in code], float)

# ---- C1 screen, committed before any RM scoring ----
art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
prob = cross_val_predict(art_model, phi, y, cv=CV, method="predict_proba")[:, 1]
index = cross_val_predict(art_model, phi, y, cv=CV, method="decision_function")
c1 = float(roc_auc_score(y, prob))
pred = ("improvement expected (C1 passes; C2 measured after scoring)"
        if c1 >= 0.65 else
        "C1 fails: label-defined slice unavailable; screen refuses "
        "slice-level repair. Committed fallback analysis: measure the RM's "
        "format-loading R2(phi->s) and the paired format/construct effects; "
        "residualization is licensed by construction (format edits are "
        "semantics-preserving), validated on the FULL population.")
committed = {"C1_screen": c1, "committed_prediction": pred,
             "gates": {"C1": 0.65, "C2_R2": 0.05, "C2_Dadv": 0.03,
                       "cert": 0.05}}
print("COMMITTED BEFORE SCORING:", json.dumps(committed))
json.dump(committed, open(OUT + ".committed", "w"), indent=1)

# ---- reward-model scoring (cached) ----
if os.path.exists(SCACHE):
    s = np.load(SCACHE)
    assert len(s) == n
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "OpenAssistant/reward-model-deberta-v3-base"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
    except Exception as e:
        print("base RM unavailable, falling back to large-v2:", e)
        name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    print("scoring with", name)
    s = np.zeros(n)
    with torch.no_grad():
        for i in range(0, n, 16):
            enc = tok(prompt[i:i + 16], code[i:i + 16], return_tensors="pt",
                      truncation=True, max_length=512, padding=True)
            s[i:i + 16] = model(**enc).logits[:, 0].numpy()
            if i % 320 == 0:
                print(f"scored {i}/{n}", flush=True)
    np.save(SCACHE, s)
s = np.asarray(s, float)

# ---- pipeline ----
srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
r2 = 1 - float((srep ** 2).sum()) / float(((s - s.mean()) ** 2).sum())


def paired_effect(u, mask_hi, mask_lo, B=2000):
    """Within-problem mean difference u[hi] - u[lo], problem-level CI."""
    qs = np.unique(qid)
    per_q = []
    for t in qs:
        hi = u[(qid == t) & mask_hi]
        lo = u[(qid == t) & mask_lo]
        if len(hi) and len(lo):
            per_q.append(hi.mean() - lo.mean())
    per_q = np.array(per_q)
    boots = [per_q[rng.integers(0, len(per_q), len(per_q))].mean()
             for _ in range(B)]
    return {"mean": float(per_q.mean()),
            "CI": [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))],
            "n_problems": int(len(per_q))}


verbose = np.isin(cell, ["correct-verbose", "buggy-verbose"])
correct = y == 1


def effects(u):
    return {
        "format_given_correct": paired_effect(u, verbose & correct,
                                              ~verbose & correct),
        "format_given_buggy": paired_effect(u, verbose & ~correct,
                                            ~verbose & ~correct),
        "construct_given_terse": paired_effect(u, correct & ~verbose,
                                               ~correct & ~verbose),
        "construct_given_verbose": paired_effect(u, correct & verbose,
                                                 ~correct & verbose)}


def cluster_auc_delta_ci(B=1000):
    qs = np.unique(qid)
    gains = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == t)[0] for t in take])
        ya = y[idx]
        if ya.min() == ya.max():
            continue
        gains.append(roc_auc_score(ya, srep[idx]) - roc_auc_score(ya, s[idx]))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def pref_acc(u):
    """P(u_correct > u_buggy) within problem x format cell."""
    wins = tot = 0
    for t in np.unique(qid):
        for fmt in (True, False):
            hi = u[(qid == t) & correct & (verbose == fmt)]
            lo = u[(qid == t) & ~correct & (verbose == fmt)]
            for a in hi:
                for b in lo:
                    wins += a > b
                    tot += 1
    return float(wins / max(1, tot))


res = {"n": n, "n_problems": int(len(np.unique(qid))),
       "C1": c1, "C2_R2_phi_to_s": r2,
       "effects_raw": effects(s), "effects_residualized": effects(srep),
       "A_full": [float(roc_auc_score(y, s)), float(roc_auc_score(y, srep))],
       "A_full_delta_clusterCI": cluster_auc_delta_ci(),
       "pref_acc": {"raw": pref_acc(s), "residualized": pref_acc(srep)},
       "cert_format_corr": [abs(float(np.corrcoef(s, index)[0, 1])),
                            abs(float(np.corrcoef(srep, index)[0, 1]))],
       "committed": committed}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
