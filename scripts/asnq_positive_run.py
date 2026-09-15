"""Second non-NLI positive attempt (round-5): ASNQ (answer-sentence selection from Natural
Questions), C1-screened at 0.747 (resampled to WikiQA-comparable per-question candidate lists;
notes/asnq_screen_resampled_2026-07-11.json). Full WikiQA recipe: s = MS MARCO cross-encoder
(MiniLM-L6, off-distribution), y = ASNQ sentence label, phi = question-blind surface channel
(answer-only TF-IDF + overlap + lengths). Reports C1, C2 (R2, Delta_adv), adversarial-slice
repair with bootstrap CI, and the channel certificate |corr(s_rep, a_hat)|.
"""
import json
import os

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

rows = json.load(open("/tmp/asnq_rows_8000.json"))
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
print(f"ASNQ n={len(y)} pos_rate={y.mean():.3f}")

cache = "/tmp/asnq_ce_s_8000.npy"
if os.path.exists(cache):
    s = np.load(cache)
    print("loaded cached scores")
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    s = np.asarray(ce.predict(list(zip(q, ans)), batch_size=64,
                              show_progress_bar=True), float)
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
c1 = roc_auc_score(y, art)
thr = np.quantile(art, 1 - y.mean())
adv = (art > thr) != (y == 1)

s_hat = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
r2 = 1 - float(np.sum((s - s_hat) ** 2)) / float(np.sum((s - s.mean()) ** 2))
A_full, A_adv = roc_auc_score(y, s), roc_auc_score(y[adv], s[adv])
delta_adv = A_full - A_adv
print(f"C1={c1:.3f}  C2: R2(phi->s)={r2:.3f}  Delta_adv={delta_adv:.3f}  "
      f"(A_full={A_full:.3f} A_adv={A_adv:.3f}, n_adv={int(adv.sum())})")

s_rep = s - s_hat
Ar_full, Ar_adv = roc_auc_score(y, s_rep), roc_auc_score(y[adv], s_rep[adv])
gains = []
idx = np.where(adv)[0]
for _ in range(2000):
    b = BOOT.choice(idx, len(idx))
    if len(np.unique(y[b])) < 2:
        continue
    gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
g = np.sort(gains)
ci = [round(float(g[int(.025 * len(g))]), 3), round(float(g[int(.975 * len(g))]), 3)]
kraw = abs(float(np.corrcoef(s, art)[0, 1]))
krep = abs(float(np.corrcoef(s_rep, art)[0, 1]))
print(f"repair: A_adv {A_adv:.3f} -> {Ar_adv:.3f}  (gain {Ar_adv-A_adv:+.3f}, CI {ci})")
print(f"        A_full {A_full:.3f} -> {Ar_full:.3f}")
print(f"certificate: |corr(s,a_hat)| {kraw:.3f} -> {krep:.3f}  [closed if <=0.05]")

out = {"n": len(y), "pos_rate": float(y.mean()), "C1": round(c1, 3),
       "C2": {"R2_phi_to_s": round(r2, 3), "Delta_adv": round(delta_adv, 3)},
       "A_full": [round(A_full, 3), round(Ar_full, 3)],
       "A_adv": [round(A_adv, 3), round(Ar_adv, 3)],
       "adv_gain_ci": ci, "n_adv": int(adv.sum()),
       "kappa": [round(kraw, 3), round(krep, 3)]}
json.dump(out, open(os.path.join(os.path.dirname(__file__),
          "../notes/asnq_positive_2026-07-11.json"), "w"), indent=1)
print("saved notes/asnq_positive_2026-07-11.json")
