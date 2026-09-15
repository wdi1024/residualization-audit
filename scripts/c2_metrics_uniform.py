"""Uniform-recipe C2 metrics for the scope table (fix of c2_metrics_negatives).

Same operator as Eq.(1) of the paper for every setting: plain Ridge(alpha=1.0)
on phi, 5-fold shuffled cross-fitting. MNLI score column re-derived from the
cached class-probability matrix exactly as cross_nli_run.py does (argmax-AUC
column), fixing the orientation of the *_s.npy cache.
Outputs R2(phi->s) and Delta_adv for MNLI, QQP, toxicity + (from cache) SNLI, SICK.
"""
import json
import os
import re

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

CV = KFold(5, shuffle=True, random_state=0)
OUT = {}


def auc(y, x, m=None):
    if m is not None:
        y, x = y[m], x[m]
    return float(roc_auc_score(y, x))


def c2_stats(name, phi, art, y, s):
    adv = (art > 0.5) != (y == 1)
    pred = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
    row = {"C1_art_auc": auc(y, art), "A_full_raw": auc(y, s),
           "A_adv_raw": auc(y, s, adv), "Delta_adv": auc(y, s) - auc(y, s, adv),
           "R2_phi_to_s": r2, "n_adv": int(adv.sum())}
    OUT[name] = row
    print(name, json.dumps(row, indent=1))


def run_mnli():
    rng = np.random.default_rng(0)
    d = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    rows = [(r["premise"], r["hypothesis"],
             1 if r["label"] == 0 else (0 if r["label"] == 2 else None))
            for r in d]
    rows = [r for r in rows if r[2] is not None]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    if len(y) > 4200:
        sel = rng.permutation(len(y))[:4000]
        hyp = [hyp[i] for i in sel]
        y = y[sel]
    P = np.load("/tmp/cross_mnli_bert-base-uncased-snli_probs.npy")
    aucs = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
    s = P[:, int(np.argmax(aucs))]
    print("mnli per-class AUC:", [f"{a:.3f}" for a in aucs])
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    art = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y, cv=CV,
                            method="predict_proba")[:, 1]
    c2_stats("MNLI_offdist", Xh, art, y, s)


def run_qqp():
    d = load_dataset("nyu-mll/glue", "qqp", split="train")
    pos = [(r["question1"], r["question2"]) for r in d if r["label"] == 1]
    neg = [(r["question1"], r["question2"]) for r in d if r["label"] == 0]
    k = 2000
    q1 = [a for a, _ in pos[:k]] + [a for a, _ in neg[:k]]
    q2 = [b for _, b in pos[:k]] + [b for _, b in neg[:k]]
    y = np.array([1] * k + [0] * k)
    s = np.load("/tmp/qqp_s_4000.npy")

    def overlap(a, b):
        A, B_ = set(a.lower().split()), set(b.lower().split())
        return len(A & B_) / max(1, len(A | B_))
    ov = np.array([overlap(a, b) for a, b in zip(q1, q2)])[:, None]
    l1 = np.array([len(a.split()) for a in q1], float)[:, None]
    l2 = np.array([len(b.split()) for b in q2], float)[:, None]
    Xq2 = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(q2)
    phi = hstack([Xq2, csr_matrix(np.hstack([ov, l1, l2, l1 - l2]))]).tocsr()
    art = cross_val_predict(make_pipeline(StandardScaler(with_mean=False),
                                          LogisticRegression(max_iter=1000)),
                            Xq2, y, cv=CV, method="predict_proba")[:, 1]
    c2_stats("QQP_indist", phi, art, y, s)


IDENT = ["black", "white", "muslim", "jewish", "christian", "gay", "lesbian",
         "homosexual", "trans", "transgender", "women", "woman", "female",
         "men", "man", "male", "immigrant", "mexican", "islam", "muslims",
         "jews", "gays", "queer", "asian", "african", "catholic", "atheist",
         "feminist"]


def run_tox():
    N = 4000
    d = load_dataset("google/civil_comments", split="train", streaming=True)
    pos, neg = [], []
    for r in d:
        t = r["text"]
        tox = r["toxicity"]
        if tox is None or not t.strip():
            continue
        lab = 1 if tox >= 0.5 else (0 if tox <= 0.1 else None)
        if lab is None:
            continue
        (pos if lab == 1 else neg).append(t)
        if len(pos) >= N // 2 and len(neg) >= N // 2:
            break
    texts = pos[:N // 2] + neg[:N // 2]
    y = np.array([1] * (N // 2) + [0] * (N // 2))
    s = np.load("/tmp/tox_s_4000.npy")

    def ident_counts(t):
        tl = t.lower()
        return [len(re.findall(r"\b" + re.escape(w) + r"\b", tl)) for w in IDENT]
    ic = np.array([ident_counts(t) for t in texts], float)
    ln = np.array([len(t.split()) for t in texts], float)[:, None]
    phi = csr_matrix(np.hstack([ic, ln, ic.sum(1, keepdims=True)]))
    art = cross_val_predict(make_pipeline(StandardScaler(with_mean=False),
                                          LogisticRegression(max_iter=1000)),
                            phi, y, cv=CV, method="predict_proba")[:, 1]
    c2_stats("Toxicity_indist", phi, art, y, s)


if __name__ == "__main__":
    for fn in (run_mnli, run_qqp, run_tox):
        try:
            fn()
        except Exception as e:
            OUT[fn.__name__] = {"error": str(e)}
            print(fn.__name__, "FAILED:", e)
    dst = os.path.join(os.path.dirname(__file__),
                       "../notes/c2_metrics_uniform_2026-07-10.json")
    with open(dst, "w") as f:
        json.dump(OUT, f, indent=1)
    print("wrote", dst)
