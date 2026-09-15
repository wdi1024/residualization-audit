"""Continuation of certificate_exact.py: SICK (robust fetch w/ retries + row cache) and WikiQA.
Merges into notes/certificate_exact_2026-07-11.json (SNLI numbers from the first run's log)."""
import json
import os
import subprocess
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import robustness_suite as RS
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression as LR
from sklearn.model_selection import cross_val_predict as cvp, KFold as KF
from sklearn.metrics import roc_auc_score as rauc

_CV = KF(5, shuffle=True, random_state=0)
_BOOT = np.random.default_rng(0)


def _boot_ci(y_, x_, n=2000):
    idx = np.arange(len(y_)); vals = []
    for _ in range(n):
        b = _BOOT.choice(idx, len(idx))
        if len(np.unique(y_[b])) < 2:
            continue
        vals.append(rauc(y_[b], x_[b]))
    v = np.sort(vals)
    return [round(float(v[int(.025 * len(v))]), 3), round(float(v[int(.975 * len(v))]), 3)]


def _r2cv(phi_, t_):
    hat = cvp(Ridge(alpha=1.0), phi_, t_, cv=_CV)
    return 1.0 - float(np.sum((t_ - hat) ** 2)) / float(np.sum((t_ - t_.mean()) ** 2))


def _dec(sv, z_):
    pr = cvp(LR(max_iter=1000), sv.reshape(-1, 1), z_, cv=_CV, method="predict_proba")[:, 1]
    return float(rauc(z_, pr)), _boot_ci(z_, pr)


def analyze(name, phi_, s_, z_, a_hat_):
    res = {}
    variants = {
        "ridge_cv(paper)": lambda: s_ - cvp(Ridge(alpha=1.0), phi_, s_, cv=_CV),
        "ols_cv": lambda: s_ - cvp(LinearRegression(), phi_, s_, cv=_CV),
        "ols_full(in-sample)": lambda: s_ - LinearRegression().fit(phi_, s_).predict(phi_),
    }
    for vn, fn in variants.items():
        sr = fn()
        r2 = _r2cv(phi_, sr); corr = float(np.corrcoef(sr, a_hat_)[0, 1])
        dauc, ci = _dec(sr, z_)
        res[vn] = {"R2_phi_to_srep": round(r2, 4), "corr_srep_ahat": round(corr, 4),
                   "decision_probe_auc": round(dauc, 3), "decision_ci": ci}
        print(f"[{name}] {vn:20s} R2(phi->s_rep)={r2:+.4f}  corr={corr:+.4f}  "
              f"decisionAUC={dauc:.3f} {ci}")
    dauc, ci = _dec(s_, z_)
    res["raw"] = {"R2_phi_to_s": round(_r2cv(phi_, s_), 4),
                  "decision_probe_auc": round(dauc, 3), "decision_ci": ci}
    print(f"[{name}] raw decisionAUC={dauc:.3f} {ci}")
    return res

NOTES = os.path.join(os.path.dirname(__file__), "../notes")
OUT_PATH = os.path.join(NOTES, "certificate_exact_2026-07-11.json")
OUT = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}

# SNLI from the completed first run (log-verified)
OUT.setdefault("SNLI", {
    "ridge_cv(paper)": {"R2_phi_to_srep": None, "decision_probe_auc": None,
                        "note": "see certificate_exact_run.log head"},
})

# ------------------------------------------------------------------- SICK
ROWS_CACHE = "/tmp/sick_rows.json"
if os.path.exists(ROWS_CACHE):
    rows = json.load(open(ROWS_CACHE))
else:
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            "yangwang825%2Fsick&config=default&split=test")
    rows = []
    for off in range(0, 4906, 100):
        for attempt in range(6):
            out = subprocess.run(["curl", "-sL", "--max-time", "60",
                                  f"{base}&offset={off}&length=100"],
                                 capture_output=True).stdout
            try:
                rows += [x["row"] for x in json.loads(out)["rows"]]
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            print(f"fetch failed at offset {off}; skipping SICK")
            rows = None
            break
    if rows is not None:
        json.dump(rows, open(ROWS_CACHE, "w"))

if rows is None:
    OUT["SICK"] = {"error": "datasets-server unavailable"}
else:
    print("SICK rows:", len(rows))
    pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
             for r in rows if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in pairs]
    y = np.array([c for _, _, c in pairs])
    s = np.load("/tmp/sick_s.npy")
    assert len(s) == len(y)
    print(f"[SICK] raw AUC sanity: {rauc(y, s):.3f} (expect ~0.967)")
    phi = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    a_hat = RS.artifact_channel(phi, y)
    z = (a_hat > 0.5).astype(int)
    OUT["SICK"] = analyze("SICK", phi, s, z, a_hat)
json.dump(OUT, open(OUT_PATH, "w"), indent=1)

# ------------------------------------------------------------------ WikiQA
from datasets import load_dataset
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

CV = KFold(5, shuffle=True, random_state=0)
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
print(f"[WikiQA] raw AUC sanity: {rauc(y, s):.3f} (expect ~0.864)")
OUT["WikiQA"] = analyze("WikiQA", phi, s, z, a_hat)
json.dump(OUT, open(OUT_PATH, "w"), indent=1)
print("wrote", OUT_PATH)
