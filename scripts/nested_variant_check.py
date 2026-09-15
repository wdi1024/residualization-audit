"""Isolate WHY SNLI fails the fully-nested protocol: slice-selection bias vs
small-n repair. Middle variant: the slice on eval-half B is defined by an
artifact predictor trained only on half A (no selection bias), but the repair
residual is the ORIGINAL full-data cross-fitted one (same operator output the
paper reports). If the gain survives here but not in the fully-nested run,
the fully-nested failure is a small-n repair effect; if it dies here too, the
pooled SNLI gain was inflated by slice selection.
"""
import json
import os
import subprocess

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
RNG = np.random.default_rng(0)
OUT = {}


def auc(y, x, m=None):
    if m is not None:
        y, x = y[m], x[m]
    return float(roc_auc_score(y, x))


def load_snli():
    d = load_dataset("stanfordnlp/snli", split="validation")
    rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load("/tmp/snli_s_2400.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return Xh, LogisticRegression(max_iter=1000), Xh, y, s, 0.5


def load_sick():
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            "yangwang825%2Fsick&config=default&split=test")
    rows = []
    for off in range(0, 4906, 100):
        out = subprocess.run(["curl", "-sL", "--max-time", "60",
                              f"{base}&offset={off}&length=100"],
                             capture_output=True, check=True).stdout
        rows += [x["row"] for x in json.loads(out)["rows"]]
    pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
             for r in rows if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in pairs]
    y = np.array([c for _, _, c in pairs])
    s = np.load("/tmp/sick_s.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return Xh, LogisticRegression(max_iter=1000), Xh, y, s, 0.5


def load_wikiqa():
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
    art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return phi, art_model, dense, y, s, "rate"


for name, loader in (("SNLI", load_snli), ("SICK", load_sick),
                     ("WikiQA", load_wikiqa)):
    phi, art_model, art_X, y, s, thr_mode = loader()
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)  # full-data op
    idx = RNG.permutation(len(y))
    halves = (idx[: len(y) // 2], idx[len(y) // 2:])
    rows = []
    for tr, ev in (halves, halves[::-1]):
        art_model.fit(art_X[tr], y[tr])
        pred_ev = art_model.predict_proba(art_X[ev])[:, 1]
        thr = 0.5 if thr_mode == 0.5 else np.quantile(pred_ev, 1 - y[ev].mean())
        adv = (pred_ev > thr) != (y[ev] == 1)
        rows.append({"n_adv": int(adv.sum()),
                     "A_adv_raw": auc(y[ev], s[ev], adv),
                     "A_adv_rep": auc(y[ev], s_rep[ev], adv),
                     "gain": auc(y[ev], s_rep[ev], adv) - auc(y[ev], s[ev], adv)})
    OUT[name] = rows
    print(name, json.dumps(rows, indent=1))

dst = os.path.join(os.path.dirname(__file__),
                   "../notes/nested_variant_2026-07-11.json")
with open(dst, "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote", dst)
