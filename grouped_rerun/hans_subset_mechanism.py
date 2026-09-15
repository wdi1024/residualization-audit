"""Per-heuristic mechanism analysis for the HANS repair outcomes (internal
review round 5, item E1).

Question: the subsequence subset has subset-level R2(phi->s) ~ 0, yet repair
moves its AUC by -0.200. How can a subtraction that explains no subset
variance move a rank statistic that much?

Answer, measured here: (1) the raw score is saturated on that subset
(median 0.994, q10 0.946, sd 0.138 -- a spike at ~1 with a thin left tail),
so the subtracted component's sd (~0.02) is comparable to the score
variation within the saturated bulk and freely reorders it; (2) the sign of
the effect is set by the subset-level correlation of the subtracted
component with the construct label: corr(g_hat, y) = +0.31 on subsequence
(subtracting removes construct signal -> -0.200), -0.28 on lexical overlap
(subtracting removes anti-construct signal -> +0.104), ~0.00 on constituent
(pure saturation reshuffle -> -0.064). Subset R2 is a variance statistic
and does not see either effect.

Requires the HANS evaluation file (public):
https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt
at /tmp/hans_eval.txt, and the cached scores in ../cache/.
"""
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
OUT = os.path.join(HERE, "hans_subset_mechanism.json")

rows = [l.rstrip("\n").split("\t") for l in open("/tmp/hans_eval.txt")]
hdr = rows[0]
ix = {c: i for i, c in enumerate(hdr)}
data = rows[1:]
prem = [r[ix["sentence1"]] for r in data]
hyp = [r[ix["sentence2"]] for r in data]
gold = [r[ix["gold_label"]] for r in data]
heur = [r[ix["heuristic"]] for r in data]
y_all = np.array([1 if g == "entailment" else 0 for g in gold])

# identical stratified subsample to hans_positive_run.py (seed 0, N=8000)
rng = np.random.default_rng(0)
idx = np.arange(len(data))
N = 8000
keys = np.array([f"{h}|{g}" for h, g in zip(heur, gold)])
per = {}
for k in np.unique(keys):
    pool = idx[keys == k]
    per[k] = rng.choice(pool, min(len(pool), N // len(np.unique(keys))), replace=False)
idx = np.sort(np.concatenate(list(per.values())))
prem = [prem[i] for i in idx]
hyp = [hyp[i] for i in idx]
heur = np.array([heur[i] for i in idx])
y = y_all[idx]
s = np.load(f"{CACHE}/hans_s_f93c1f27.npy")
assert len(s) == len(y)


def overlap_ratio(p, h):
    hs = h.lower().split()
    ps = set(p.lower().split())
    return sum(w in ps for w in hs) / max(1, len(hs))


ov = np.array([overlap_ratio(p, h) for p, h in zip(prem, hyp)])
lp = np.array([len(p.split()) for p in prem])
lh = np.array([len(h.split()) for h in hyp])
phi = csr_matrix(np.c_[ov, lp, lh, lp - lh])
CV = KFold(5, shuffle=True, random_state=0)
pred = cross_val_predict(make_pipeline(StandardScaler(with_mean=False),
                                       Ridge(alpha=1.0)), phi, s, cv=CV)
srep = s - pred

out = {"global": {"raw": round(float(roc_auc_score(y, s)), 4),
                  "rep": round(float(roc_auc_score(y, srep)), 4)}}
for h in np.unique(heur):
    m = heur == h
    r2 = 1 - float(((s[m] - pred[m]) ** 2).sum()) / float(((s[m] - s[m].mean()) ** 2).sum())
    out[h] = {
        "n": int(m.sum()),
        "auc": [round(float(roc_auc_score(y[m], s[m])), 4),
                round(float(roc_auc_score(y[m], srep[m])), 4)],
        "subset_R2": round(r2, 4),
        "s_median": round(float(np.median(s[m])), 4),
        "s_q10": round(float(np.quantile(s[m], 0.10)), 4),
        "s_sd": round(float(s[m].std()), 4),
        "pred_sd": round(float(pred[m].std()), 4),
        "corr_pred_y": round(float(np.corrcoef(pred[m], y[m])[0, 1]), 4),
    }
    print(h, json.dumps(out[h]))

json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
