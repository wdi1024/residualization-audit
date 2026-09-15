"""Round-5 experiment 1: does SNLI survive the fully-nested protocol once
p >> n is removed by dimensionality reduction?

Diagnosis under test: SNLI's fully-nested failure (gains -0.019/-0.003 at
n_eval=1200, p=8000) is a ridge-degradation effect, not a selection artifact.
If true, projecting the TF-IDF phi to k dims via TruncatedSVD (fit within the
evaluation half only, no leakage) should restore the nested gain.

Reports, for k in {64, 128, 256} plus the p=8000 baseline:
  pooled gain (full-data operator, OOF slice — the paper's main protocol)
  fully-nested gains (slice predictor trained on other half; SVD + repair
  cross-fitted within the evaluation half)
"""
import json
import os

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

CV = KFold(5, shuffle=True, random_state=0)
RNG = np.random.default_rng(0)

d = load_dataset("stanfordnlp/snli", split="validation")
rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
        for r in d if r["label"] in (0, 2)][:2400]
hyp = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
s = np.load("/tmp/snli_s_2400.npy")
Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)


def auc(yv, x, m=None):
    if m is not None:
        yv, x = yv[m], x[m]
    return float(roc_auc_score(yv, x))


def phi_k(X, k):
    if k is None:
        return X
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(X)


OUT = {}
for k in (None, 64, 128, 256):
    key = "p8000" if k is None else f"svd{k}"
    # ---- pooled (paper protocol)
    art = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y, cv=CV,
                            method="predict_proba")[:, 1]
    adv = (art > 0.5) != (y == 1)
    P = phi_k(Xh, k)
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), P, s, cv=CV)
    pooled = {"A_adv_raw": auc(y, s, adv), "A_adv_rep": auc(y, s_rep, adv),
              "gain": auc(y, s_rep, adv) - auc(y, s, adv)}

    # ---- fully nested (slice model on other half; SVD+repair inside eval half)
    idx = RNG.permutation(len(y))
    halves = (idx[: len(y) // 2], idx[len(y) // 2:])
    nested = []
    for tr, ev in (halves, halves[::-1]):
        m = LogisticRegression(max_iter=1000).fit(Xh[tr], y[tr])
        pred_ev = m.predict_proba(Xh[ev])[:, 1]
        adv_ev = (pred_ev > 0.5) != (y[ev] == 1)
        P_ev = phi_k(Xh[ev], k)
        s_ev = s[ev]
        s_rep_ev = s_ev - cross_val_predict(Ridge(alpha=1.0), P_ev, s_ev, cv=CV)
        nested.append({"n_adv": int(adv_ev.sum()),
                       "gain": auc(y[ev], s_rep_ev, adv_ev)
                       - auc(y[ev], s_ev, adv_ev)})
    OUT[key] = {"pooled": pooled, "nested": nested}
    print(key, json.dumps(OUT[key], indent=1))

dst = os.path.join(os.path.dirname(__file__),
                   "../notes/snli_svd_nested_2026-07-11.json")
with open(dst, "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote", dst)
