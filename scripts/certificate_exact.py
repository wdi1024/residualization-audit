"""Reviewer round-4 must-fix #1: reconcile the linear-certificate claim with App-G numbers.

Distinction under test: population-OLS residualization guarantees zero correlation between
s_rep and EVERY LINEAR FUNCTION OF phi. The App-G probe instead predicts the artifact-only
model's THRESHOLDED DECISION z = 1[a_hat > t] from s_rep — a nonlinear (threshold) target that
sits OUTSIDE the guarantee class. If SICK's 0.648 comes from ridge shrinkage / cross-fit slack,
an exact OLS projection should close it; if it comes from the threshold nonlinearity, the
guaranteed-class metrics (linear R^2(phi -> s_rep), corr(s_rep, a_hat)) will be ~0 for every
residualizer while the decision-AUC stays where it is.

Per setting (SNLI / SICK / WikiQA) x residualizer (Ridge-CV [paper], OLS-CV, OLS-full in-sample
projection):
  guaranteed-class:  cross-fitted linear R^2(phi -> s_rep); |corr(s_rep, a_hat_cv)|
  outside-guarantee: linear-probe AUC predicting z from s_rep (App-G protocol) + bootstrap CI

Output: notes/certificate_exact_2026-07-11.json + stdout table.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

import robustness_suite as RS  # loaders + helpers (heavy work is under __main__)

CV = KFold(5, shuffle=True, random_state=0)
BOOT = np.random.default_rng(0)
OUT = {}


def boot_ci(y, x, n=2000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = BOOT.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], x[b]))
    v = np.sort(vals)
    return [round(float(v[int(0.025 * len(v))]), 3),
            round(float(v[int(0.975 * len(v))]), 3)]


def linear_r2_cv(phi, target):
    hat = cross_val_predict(Ridge(alpha=1.0), phi, target, cv=CV)
    ss_res = float(np.sum((target - hat) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def decision_probe(s_vec, z):
    X = s_vec.reshape(-1, 1)
    pr = cross_val_predict(LogisticRegression(max_iter=1000), X, z,
                           cv=CV, method="predict_proba")[:, 1]
    return float(roc_auc_score(z, pr)), boot_ci(z, pr)


def analyze(name, phi, s, z, a_hat):
    res = {}
    variants = {
        "ridge_cv(paper)": lambda: s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV),
        "ols_cv": lambda: s - cross_val_predict(LinearRegression(), phi, s, cv=CV),
        "ols_full(in-sample)": lambda: s - LinearRegression().fit(phi, s).predict(phi),
    }
    for vn, fn in variants.items():
        sr = fn()
        r2 = linear_r2_cv(phi, sr)
        corr = float(np.corrcoef(sr, a_hat)[0, 1])
        dauc, ci = decision_probe(sr, z)
        res[vn] = {"R2_phi_to_srep": round(r2, 4),
                   "corr_srep_ahat": round(corr, 4),
                   "decision_probe_auc": round(dauc, 3), "decision_ci": ci}
        print(f"[{name}] {vn:20s} R2(phi->s_rep)={r2:+.4f}  "
              f"corr(s_rep,a_hat)={corr:+.4f}  decisionAUC={dauc:.3f} {ci}")
    # raw reference
    dauc, ci = decision_probe(s, z)
    r2raw = linear_r2_cv(phi, s)
    res["raw"] = {"R2_phi_to_s": round(r2raw, 4),
                  "decision_probe_auc": round(dauc, 3), "decision_ci": ci}
    print(f"[{name}] raw                  R2(phi->s)={r2raw:+.4f}  "
          f"decisionAUC={dauc:.3f} {ci}")
    return res


# ---------------------------------------------------------------- NLI settings
for name, loader in (("SNLI", RS.load_snli), ("SICK", RS.load_sick)):
    hyp, y, s = loader()
    phi = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    a_hat = RS.artifact_channel(phi, y)
    z = (a_hat > 0.5).astype(int)
    OUT[name] = analyze(name, phi, s, z, a_hat)

# ---------------------------------------------------------------- WikiQA
from datasets import load_dataset
from scipy.sparse import hstack, csr_matrix

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
s = np.load("/tmp/wikiqa_ce_s_8000.npy")


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()
a_hat = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                          dense, y, cv=CV, method="predict_proba")[:, 1]
thr = np.quantile(a_hat, 1 - y.mean())
z = (a_hat > thr).astype(int)
print(f"[WikiQA] n={len(y)} raw-AUC sanity: {roc_auc_score(y, s):.3f} (expect ~0.864)")
OUT["WikiQA"] = analyze("WikiQA", phi, s, z, a_hat)

dst = os.path.join(os.path.dirname(__file__), "../notes/certificate_exact_2026-07-11.json")
json.dump(OUT, open(dst, "w"), indent=1)
print("wrote", dst)
