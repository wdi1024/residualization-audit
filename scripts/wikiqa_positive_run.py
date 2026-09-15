"""Non-NLI positive attempt: WikiQA answer selection (ranking/QA domain).

Screen result (non_nli_screen.py, 2026-07-10): overlap+lengths artifact-only
AUC = 0.747, answer-only TF-IDF = 0.677 -> C1 passes with the strongest
linear channel found outside NLI.

Recipe mirrors SNLI/SICK: scorer trained on a DIFFERENT dataset (MS MARCO
cross-encoder ms-marco-MiniLM-L-6-v2) applied to WikiQA sentence selection;
construct y = crowdsourced answer label (independent of the MS MARCO
pipeline); artifact phi = question-blind/dense surface channel (answer-only
TF-IDF + overlap + lengths); repair = cross-fit Ridge residual.

WikiQA's positive rate is 0.049, so the artifact-adversarial slice uses a
rate-calibrated threshold on the artifact predictor (predicted-positive rate
matched to base rate), exactly as in the linear-SVM variant of the appendix;
a 0.5 threshold would make the slice single-class.
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
print(f"WikiQA n={len(y)} pos_rate={y.mean():.3f}")

# ---------- score s = MS MARCO cross-encoder relevance (off-distribution)
cache = "/tmp/wikiqa_ce_s_8000.npy"
if os.path.exists(cache):
    s = np.load(cache)
    print("loaded cached scores")
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    s = ce.predict(list(zip(q, ans)), batch_size=64, show_progress_bar=True)
    s = np.asarray(s, float)
    np.save(cache, s)

# ---------- artifact phi
def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()

# artifact channel for the slice = the dense screen winner (0.747)
art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        dense, y, cv=CV, method="predict_proba")[:, 1]
c1 = roc_auc_score(y, art)
thr = np.quantile(art, 1 - y.mean())        # rate-calibrated (base rate 0.049)
adv = (art > thr) != (y == 1)
full = np.ones(len(y), bool)

# ---------- repair
s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)


def auc(m, x):
    return roc_auc_score(y[m], x[m])


# ---------- C2 statistics (uniform recipe)
pred = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
delta_adv = auc(full, s) - auc(adv, s)

# ---------- certificate: linear floor on the artifact decision
z = (art > thr).astype(int)


def floor_linear(x):
    p = cross_val_predict(LogisticRegression(max_iter=1000),
                          x.reshape(-1, 1), z, cv=CV, method="predict_proba")[:, 1]
    return roc_auc_score(z, p)


# ---------- bootstrap CI on the adversarial-slice gain
idx_adv = np.where(adv)[0]
gains = []
for _ in range(1000):
    b = rng.choice(idx_adv, len(idx_adv), replace=True)
    if len(set(y[b])) < 2:
        continue
    gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
lo, hi = np.percentile(gains, [2.5, 97.5])
g_full = []
for _ in range(1000):
    b = rng.integers(0, len(y), len(y))
    if len(set(y[b])) < 2:
        continue
    g_full.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
flo, fhi = np.percentile(g_full, [2.5, 97.5])

out = {
    "n": len(y), "pos_rate": float(y.mean()),
    "C1_dense": c1,
    "C2_R2_phi_to_s": r2, "C2_Delta_adv": float(delta_adv),
    "A_full_raw": auc(full, s), "A_full_rep": auc(full, s_rep),
    "A_adv_raw": auc(adv, s), "A_adv_rep": auc(adv, s_rep),
    "n_adv": int(adv.sum()),
    "adv_gain": auc(adv, s_rep) - auc(adv, s), "adv_CI": [float(lo), float(hi)],
    "full_gain": auc(full, s_rep) - auc(full, s), "full_CI": [float(flo), float(fhi)],
    "floor_raw": floor_linear(s), "floor_rep": floor_linear(s_rep),
}
print(json.dumps(out, indent=1))
dst = os.path.join(os.path.dirname(__file__), "../notes/wikiqa_result_2026-07-10.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", dst)
