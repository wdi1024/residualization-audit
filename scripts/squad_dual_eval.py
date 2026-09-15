"""Round-9: independent y_eval for the held-out SQuAD setting.

SQuAD validation provides multiple annotator answer spans per question.
  y_slice = containment of the FIRST annotator's span (identical to the
            held-out run: artifact model, slice, C2, repair all use only this)
  y_eval  = containment of ANY of the REMAINING annotators' spans
            (disjoint annotators; used for nothing but final evaluation)

Rows whose question has fewer than 2 distinct annotator spans are excluded
from the y_eval evaluation (but retained in the pipeline, which never sees
y_eval). Writes ../notes/squad_dual_eval.json.
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
SCACHE = os.path.join(HERE, "..", "cache", "squad_ce_s_8000.npy")


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


from datasets import load_dataset
d = load_dataset("squad", split="validation")
rows, qid, alt_answers = [], [], []
seen_q = 0
for r in d:
    sents = sent_split(r["context"])
    if not (3 <= len(sents) <= 25):
        continue
    ans = r["answers"]["text"][0]
    pos = [i for i, s in enumerate(sents) if ans in s]
    if len(pos) != 1:
        continue
    alts = [t for t in r["answers"]["text"][1:]]
    for i, s in enumerate(sents):
        rows.append((r["question"], s, 1 if i == pos[0] else 0))
        qid.append(seen_q)
        alt_answers.append(alts)
    seen_q += 1
    if len(rows) >= 8000:
        break
rows = rows[:8000]
qid = np.array(qid[:8000])
alt_answers = alt_answers[:8000]
q = [r[0] for r in rows]
ans_s = [r[1] for r in rows]
y1 = np.array([r[2] for r in rows])
# y_eval: any remaining annotator span contained in the sentence
has_alt = np.array([len(a) > 0 for a in alt_answers])
y2 = np.array([1 if any(t in s for t in a) else 0
               for s, a in zip(ans_s, alt_answers)])
print(f"n={len(y1)} questions={seen_q} rows_with_alt={int(has_alt.sum())} "
      f"y1y2_agree_on_alt={float((y1[has_alt]==y2[has_alt]).mean()):.3f}")

ov = np.array([overlap(a, b) for a, b in zip(q, ans_s)])[:, None]
la = np.array([len(x.split()) for x in ans_s], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans_s)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()

if os.path.exists(SCACHE):
    s = np.load(SCACHE)
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    s = np.asarray(ce.predict(list(zip(q, ans_s)), batch_size=64), float)
    np.save(SCACHE, s)

art = cross_val_predict(make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=1000)),
                        dense, y1, cv=CV, method="predict_proba")[:, 1]
thr = np.quantile(art, 1 - y1.mean())
adv = (art > thr) != (y1 == 1)
srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)

m = adv & has_alt
res = {"n": len(y1), "n_adv": int(adv.sum()), "n_adv_with_alt": int(m.sum()),
       "y1_y2_agreement_on_alt_rows": float((y1[has_alt] == y2[has_alt]).mean()),
       "slice_gain_y1_same_label": float(
           roc_auc_score(y1[adv], srep[adv]) - roc_auc_score(y1[adv], s[adv])),
       "slice_gain_y1_on_alt_subset": float(
           roc_auc_score(y1[m], srep[m]) - roc_auc_score(y1[m], s[m])),
       "slice_gain_y2_independent": float(
           roc_auc_score(y2[m], srep[m]) - roc_auc_score(y2[m], s[m])),
       "slice_AUC_y2": [float(roc_auc_score(y2[m], s[m])),
                        float(roc_auc_score(y2[m], srep[m]))]}


def cluster_ci(yv, mask, B=1000):
    qs = np.unique(qid[mask])
    gains = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where((qid == t) & mask)[0] for t in take])
        ya, sa, ra = yv[idx], s[idx], srep[idx]
        if ya.min() == ya.max():
            continue
        gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


res["gain_y2_clusterCI"] = cluster_ci(y2, m)
json.dump(res, open(os.path.join(HERE, "..", "notes",
                                 "squad_dual_eval.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
