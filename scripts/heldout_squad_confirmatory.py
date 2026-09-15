"""Held-out confirmatory run (round-7 review): SQuAD answer-sentence selection.

Everything is FROZEN before any scoring, matching the paper's recipe verbatim:
  - candidate construction: per-question sentence candidates, positive = the
    sentence containing the answer span (ASNQ/WikiQA-style lists);
  - phi: answer-only TF-IDF (8000, 1-2gram) + lengths + question-candidate
    overlap; dense screen features = [overlap, len_a, len_q, len_a-len_q];
  - gates: C1 >= 0.65 (dense artifact-only AUC), C2: R2(phi->s) >= 0.05 and
    Delta_adv >= 0.03; certificate <= 0.05 on the linear index; ridge alpha=1,
    KFold(5, shuffle, seed 0); slice threshold rate-calibrated.
  - scorer: cross-encoder/ms-marco-MiniLM-L-6-v2, off-distribution, scored
    ONCE after the C1 screen is committed.

Single-shot protocol: the C1 screen and the gate-based prediction are printed
and saved BEFORE the scorer runs; nothing is tuned afterwards.

Also reports: query-clustered bootstrap CI, per-query AUC and MRR,
phi-only flagged subgroup (z=1), overlap-quartile observable slices,
answer-only vs overlap-only feature-split repair, and bootstrap CIs for the
certificate correlations.

Writes ../notes/heldout_squad_confirmatory.json.
"""
import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)
OUT = os.path.join(HERE, "..", "notes", "heldout_squad_confirmatory.json")


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


# ---------------- build candidates (no scoring involved) ----------------
from datasets import load_dataset
d = load_dataset("squad", split="validation")
rows, qid = [], []
seen_q = 0
for r in d:
    sents = sent_split(r["context"])
    if not (3 <= len(sents) <= 25):
        continue
    ans = r["answers"]["text"][0]
    pos = [i for i, s in enumerate(sents) if ans in s]
    if len(pos) != 1:
        continue
    for i, s in enumerate(sents):
        rows.append((r["question"], s, 1 if i == pos[0] else 0))
        qid.append(seen_q)
    seen_q += 1
    if len(rows) >= 8000:
        break
rows = rows[:8000]
qid = np.array(qid[:8000])
q = [r[0] for r in rows]
ans = [r[1] for r in rows]
y = np.array([r[2] for r in rows])
print(f"built n={len(y)} candidates, {seen_q} questions, pos_rate={y.mean():.3f}")

ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi_full = hstack([Xa, csr_matrix(dense)]).tocsr()
phi_ans = hstack([Xa, csr_matrix(la)]).tocsr()
phi_ov = csr_matrix(np.hstack([ov, la - lq]))

# ---------------- C1 screen + committed prediction (BEFORE scoring) -----
art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
prob = cross_val_predict(art_model, dense, y, cv=CV, method="predict_proba")[:, 1]
index = cross_val_predict(art_model, dense, y, cv=CV,
                          method="decision_function")
c1 = roc_auc_score(y, prob)
pred = "repair expected (C1 passes; C2 to be measured)" if c1 >= 0.65 else \
       "refusal expected (C1 fails)"
committed = {"C1_screen": float(c1), "committed_prediction": pred,
             "gates": {"C1": 0.65, "C2_R2": 0.05, "C2_Dadv": 0.03,
                       "cert": 0.05}}
print("COMMITTED BEFORE SCORING:", json.dumps(committed))
json.dump(committed, open(OUT + ".committed", "w"), indent=1)

# ---------------- score ONCE ----------------
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
s = np.asarray(ce.predict(list(zip(q, ans)), batch_size=64,
                          show_progress_bar=False), float)

# ---------------- frozen pipeline ----------------
def repair(ph):
    return s - cross_val_predict(Ridge(alpha=1.0), ph, s, cv=CV)

srep = repair(phi_full)
thr = np.quantile(prob, 1 - y.mean())
flagged = prob > thr
adv = (prob > thr) != (y == 1)
r2 = 1 - float(((s - cross_val_predict(Ridge(alpha=1.0), phi_full, s, cv=CV)) ** 2).sum()) \
       / float(((s - s.mean()) ** 2).sum())
A = lambda m, u: float(roc_auc_score(y[m], u[m])) if y[m].min() != y[m].max() else None
full = np.ones(len(y), bool)
dadv = A(full, s) - A(adv, s)

def cluster_ci(mask, B=1000):
    qs = np.unique(qid)
    gains = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == t)[0] for t in take])
        m = mask[idx]
        ya, sa, ra = y[idx][m], s[idx][m], srep[idx][m]
        if ya.min() == ya.max():
            continue
        gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]

def per_query(u):
    aucs, rr = [], []
    for uq in np.unique(qid):
        m = qid == uq
        if y[m].min() != y[m].max():
            aucs.append(roc_auc_score(y[m], u[m]))
        if y[m].sum() > 0:
            order = np.argsort(-u[m])
            rr.append(1.0 / (1 + int(np.where(y[m][order] == 1)[0][0])))
    return float(np.mean(aucs)), float(np.mean(rr))

def corr_ci(u, v, B=1000):
    vals = []
    for _ in range(B):
        ii = rng.integers(0, len(u), len(u))
        vals.append(abs(float(np.corrcoef(u[ii], v[ii])[0, 1])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

hi_ov = (ov[:, 0] >= np.quantile(ov[:, 0], 0.75))
res = {
    "n": len(y), "n_questions": int(seen_q), "pos_rate": float(y.mean()),
    "C1": float(c1), "C2_R2_phi_to_s": float(r2), "C2_Delta_adv": float(dadv),
    "gates_pass": bool(c1 >= 0.65 and r2 >= 0.05 and dadv >= 0.03),
    "A_full": [A(full, s), A(full, srep)],
    "A_slice": [A(adv, s), A(adv, srep)], "n_adv": int(adv.sum()),
    "slice_gain_clusterCI": cluster_ci(adv),
    "A_flagged_z1": [A(flagged, s), A(flagged, srep)],
    "A_high_overlap_q4": [A(hi_ov, s), A(hi_ov, srep)],
    "perquery_AUC": [per_query(s)[0], per_query(srep)[0]],
    "MRR": [per_query(s)[1], per_query(srep)[1]],
    "cert_index": [abs(float(np.corrcoef(s, index)[0, 1])),
                   abs(float(np.corrcoef(srep, index)[0, 1]))],
    "cert_index_rep_CI": corr_ci(srep, index),
    "ablation_gain": {}}
for tag, ph in [("full", phi_full), ("answer_only", phi_ans),
                ("overlap_only", phi_ov)]:
    rr_ = repair(ph)
    res["ablation_gain"][tag] = float(roc_auc_score(y[adv], rr_[adv])
                                      - roc_auc_score(y[adv], s[adv]))
res["committed"] = committed
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("saved", OUT)
