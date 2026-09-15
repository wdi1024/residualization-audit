"""Round-5 'another QA dataset': C1 screen on ASNQ (answer-sentence selection built from
Natural Questions — the reviewer's named candidate). Same artifact features as WikiQA
(question-blind: answer-only TF-IDF + overlap + lengths; C1 gate uses the dense screen
features alone, exactly like the WikiQA screen). Subsample ~8k rows question-grouped from
the validation split, rate-matched to keep the setting comparable.

Gate: C1 (artifact-only AUC over dense features) >= ~0.65 -> proceed to CE scoring.
"""
import json
import os

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

d = load_dataset("asnq", split="validation")
# group rows by question, keep questions until ~8000 rows
by_q = {}
for r in d:
    by_q.setdefault(r["question"], []).append((r["sentence"], r["label"]))
qs = list(by_q.keys())
rng.shuffle(qs)
rows = []
for qq in qs:
    for sent, lab in by_q[qq]:
        rows.append((qq, sent, int(lab)))
    if len(rows) >= 8000:
        break
rows = rows[:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
print(f"ASNQ n={len(y)} pos_rate={y.mean():.4f} questions={len(set(q))}")


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        dense, y, cv=CV, method="predict_proba")[:, 1]
c1 = roc_auc_score(y, art)
print(f"ASNQ C1 (overlap+length artifact-only AUC): {c1:.3f}  [gate ~0.65]")

out = {"dataset": "asnq(validation, question-grouped 8k)", "n": len(y),
       "pos_rate": float(y.mean()), "C1": round(float(c1), 3)}
json.dump(out, open(os.path.join(os.path.dirname(__file__),
          "../notes/asnq_screen_2026-07-11.json"), "w"), indent=1)
print("saved notes/asnq_screen_2026-07-11.json")
