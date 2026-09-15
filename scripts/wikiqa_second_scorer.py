"""Round-4 item 5: repeat WikiQA with a SECOND, architecturally different
off-distribution scorer (QNLI cross-encoder trained on SQuAD-derived QNLI,
not MS MARCO). Same phi, same y, same protocol as wikiqa_positive_run.py.
"""
import json
import os

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
rng = np.random.default_rng(0)

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])

cache = "/tmp/wikiqa_qnli_s_8000.npy"
if os.path.exists(cache):
    s = np.load(cache)
    print("loaded cached scores")
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/qnli-distilroberta-base", max_length=256)
    s = ce.predict(list(zip(q, ans)), batch_size=64, show_progress_bar=True)
    s = np.asarray(s, float)
    np.save(cache, s)

def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()

art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        dense, y, cv=CV, method="predict_proba")[:, 1]
thr = np.quantile(art, 1 - y.mean())
adv = (art > thr) != (y == 1)
full = np.ones(len(y), bool)
s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)

def auc(m, x):
    return roc_auc_score(y[m], x[m])

pred = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())

idx_adv = np.where(adv)[0]
gains = []
for _ in range(1000):
    b = rng.choice(idx_adv, len(idx_adv), replace=True)
    if len(set(y[b])) < 2:
        continue
    gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
lo, hi = np.percentile(gains, [2.5, 97.5])

out = {"scorer": "cross-encoder/qnli-distilroberta-base",
       "C2_R2_phi_to_s": r2,
       "C2_Delta_adv": float(auc(full, s) - auc(adv, s)),
       "A_full_raw": auc(full, s), "A_full_rep": auc(full, s_rep),
       "A_adv_raw": auc(adv, s), "A_adv_rep": auc(adv, s_rep),
       "n_adv": int(adv.sum()),
       "adv_gain": auc(adv, s_rep) - auc(adv, s),
       "adv_CI": [float(lo), float(hi)]}
print(json.dumps(out, indent=1))
dst = os.path.join(os.path.dirname(__file__),
                   "../notes/wikiqa_qnli_2026-07-11.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", dst)
