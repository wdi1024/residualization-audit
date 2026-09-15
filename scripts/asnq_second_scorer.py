"""Round-6 #1-ranked experiment: ASNQ scorer replication. Same rows/slice protocol as
asnq_positive_run.py (cached /tmp/asnq_rows_8000.json), scored by two further cross-encoders
with different architectures/training distributions (QNLI DistilRoBERTa, STS-B RoBERTa),
mirroring the WikiQA three-scorer replication. Reports C2, adversarial-slice gain + CI,
and the channel certificate per scorer."""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

CV = KFold(5, shuffle=True, random_state=0)
BOOT = np.random.default_rng(0)

SCORERS = {
    "qnli": "cross-encoder/qnli-distilroberta-base",
    "stsb": "cross-encoder/stsb-roberta-base",
}

rows = json.load(open("/tmp/asnq_rows_8000.json"))
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])


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
idx = np.where(adv)[0]
print(f"ASNQ n={len(y)} n_adv={int(adv.sum())} (same slice as primary run)")

OUT = {}
for key, model in SCORERS.items():
    cache = f"/tmp/asnq_{key}_s_8000.npy"
    if os.path.exists(cache):
        s = np.load(cache)
        print(f"[{key}] loaded cached scores")
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model, max_length=256)
        s = np.asarray(ce.predict(list(zip(q, ans)), batch_size=64,
                                  show_progress_bar=False), float)
        if s.ndim > 1:            # qnli head can emit two logits
            s = s[:, -1] if s.shape[1] > 1 else s.ravel()
        np.save(cache, s)
    s_hat = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    r2 = 1 - float(np.sum((s - s_hat) ** 2)) / float(np.sum((s - s.mean()) ** 2))
    s_rep = s - s_hat
    A_full, A_adv = roc_auc_score(y, s), roc_auc_score(y[adv], s[adv])
    Ar_full, Ar_adv = roc_auc_score(y, s_rep), roc_auc_score(y[adv], s_rep[adv])
    gains = []
    for _ in range(2000):
        b = BOOT.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
    g = np.sort(gains)
    ci = [round(float(g[int(.025 * len(g))]), 3), round(float(g[int(.975 * len(g))]), 3)]
    klin = [round(abs(float(np.corrcoef(s, art)[0, 1])), 3),
            round(abs(float(np.corrcoef(s_rep, art)[0, 1])), 3)]
    OUT[key] = {"scorer": model, "R2_phi_to_s": round(r2, 3),
                "A_full": [round(A_full, 3), round(Ar_full, 3)],
                "A_adv": [round(A_adv, 3), round(Ar_adv, 3)],
                "adv_gain": round(Ar_adv - A_adv, 3), "adv_gain_ci": ci,
                "R_lin": klin}
    print(f"[{key}] R2={r2:.3f}  A_adv {A_adv:.3f}->{Ar_adv:.3f} "
          f"(gain {Ar_adv-A_adv:+.3f} CI {ci})  A_full {A_full:.3f}->{Ar_full:.3f}  "
          f"R_lin {klin[0]}->{klin[1]}")

json.dump(OUT, open(os.path.join(os.path.dirname(__file__),
          "../notes/asnq_second_scorer_2026-07-11.json"), "w"), indent=1)
print("saved notes/asnq_second_scorer_2026-07-11.json")
