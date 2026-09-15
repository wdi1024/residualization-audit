"""Second held-out frozen-pipeline replication: DuoRC (SelfRC) answer-sentence
selection. Movie-plot summaries---a text domain disjoint from every prior
setting (Wikipedia, news, trivia). Protocol identical to the SQuAD run:
recipe, features, gates, slice rule, and operator frozen; the C1 screen and
gate-based prediction committed to a separate artifact BEFORE the
cross-encoder produces a single score; everything then evaluated once.

Writes ../notes/heldout_duorc_confirmatory.json (+ .committed).
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
OUT = os.path.join(HERE, "..", "notes", "heldout_duorc_confirmatory.json")
SCACHE = os.path.join(HERE, "..", "cache", "duorc_ce_s_8000.npy")


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


from datasets import load_dataset
d = load_dataset("ibm/duorc", "SelfRC", split="validation")
rows, qid = [], []
seen_q = 0
for r in d:
    if r["no_answer"] or not r["answers"]:
        continue
    sents = sent_split(r["plot"])
    if not (3 <= len(sents) <= 25):
        continue
    ans = r["answers"][0]
    if len(ans.split()) > 12 or not ans.strip():
        continue
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
ans_s = [r[1] for r in rows]
y = np.array([r[2] for r in rows])
print(f"built n={len(y)} candidates, {seen_q} questions, pos_rate={y.mean():.3f}")

ov = np.array([overlap(a, b) for a, b in zip(q, ans_s)])[:, None]
la = np.array([len(x.split()) for x in ans_s], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans_s)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()

art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
prob = cross_val_predict(art_model, dense, y, cv=CV, method="predict_proba")[:, 1]
index = cross_val_predict(art_model, dense, y, cv=CV,
                          method="decision_function")
c1 = roc_auc_score(y, prob)
pred = "improvement expected (C1 passes; C2 measured after scoring)" \
    if c1 >= 0.65 else "refusal expected (C1 fails)"
committed = {"C1_screen": float(c1), "committed_prediction": pred,
             "gates": {"C1": 0.65, "C2_R2": 0.05, "C2_Dadv": 0.03,
                       "cert": 0.05}}
print("COMMITTED BEFORE SCORING:", json.dumps(committed))
json.dump(committed, open(OUT + ".committed", "w"), indent=1)

if os.path.exists(SCACHE):
    s = np.load(SCACHE)
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    s = np.asarray(ce.predict(list(zip(q, ans_s)), batch_size=64), float)
    np.save(SCACHE, s)

srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
thr = np.quantile(prob, 1 - y.mean())
flagged = prob > thr
adv = (prob > thr) != (y == 1)
r2 = 1 - float(((srep) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
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


def corr_ci(u, v, B=1000):
    vals = []
    for _ in range(B):
        ii = rng.integers(0, len(u), len(u))
        vals.append(abs(float(np.corrcoef(u[ii], v[ii])[0, 1])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


res = {"n": len(y), "n_questions": int(seen_q), "pos_rate": float(y.mean()),
       "C1": float(c1), "C2_R2_phi_to_s": float(r2),
       "C2_Delta_adv": float(dadv),
       "gates_pass": bool(c1 >= 0.65 and r2 >= 0.05 and dadv >= 0.03),
       "A_full": [A(full, s), A(full, srep)],
       "A_slice": [A(adv, s), A(adv, srep)], "n_adv": int(adv.sum()),
       "slice_gain_clusterCI": cluster_ci(adv),
       "A_flagged_z1": [A(flagged, s), A(flagged, srep)],
       "cert_index": [abs(float(np.corrcoef(s, index)[0, 1])),
                      abs(float(np.corrcoef(srep, index)[0, 1]))],
       "cert_index_rep_CI": corr_ci(srep, index),
       "committed": committed}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
