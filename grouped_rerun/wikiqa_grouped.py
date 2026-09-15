"""WikiQA headline rerun: row-level KFold (paper) vs question-GroupKFold (leakage fix).

Identical computation to scripts/wikiqa_positive_run.py; only the CV object and
groups argument differ between the two passes. Outputs side-by-side headline
numbers so the leakage sensitivity is directly readable.
"""
import json, os
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache/wikiqa_ce_s_8000.npy")

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
s = np.load(CACHE)
assert len(s) == len(y)

# group key = question string
uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
groups = np.array([uniq[qq] for qq in q])
print(f"n={len(y)} pos_rate={y.mean():.3f} | questions={len(uniq)}")

def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()

def run(tag, cv, grp):
    kw = dict(cv=cv) if grp is None else dict(cv=cv, groups=grp)
    art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                            dense, y, method="predict_proba", **kw)[:, 1]
    c1 = roc_auc_score(y, art)
    thr = np.quantile(art, 1 - y.mean())
    adv = (art > thr) != (y == 1)
    full = np.ones(len(y), bool)
    pred = cross_val_predict(Ridge(alpha=1.0), phi, s, **kw)
    s_rep = s - pred
    r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
    def auc(m, x): return roc_auc_score(y[m], x[m])
    out = {
        "C1_dense": round(c1, 4),
        "C2_R2": round(r2, 4),
        "n_adv": int(adv.sum()),
        "A_adv_raw": round(auc(adv, s), 4), "A_adv_rep": round(auc(adv, s_rep), 4),
        "adv_gain": round(auc(adv, s_rep) - auc(adv, s), 4),
        "A_full_raw": round(auc(full, s), 4), "A_full_rep": round(auc(full, s_rep), 4),
        "full_gain": round(auc(full, s_rep) - auc(full, s), 4),
    }
    print(f"[{tag}] " + json.dumps(out))
    return out

orig = run("row-KFold (paper)", KFold(5, shuffle=True, random_state=0), None)
grpd = run("question-GroupKFold", GroupKFold(5), groups)

diff = {k: round(grpd[k] - orig[k], 4) for k in orig if isinstance(orig[k], float)}
print("\n[delta grouped - paper]", json.dumps(diff))
json.dump({"paper_cv": orig, "grouped_cv": grpd, "delta": diff},
          open(os.path.join(HERE, "wikiqa_grouped_result.json"), "w"), indent=1)
