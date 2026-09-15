"""Query-clustered bootstrap intervals on the WikiQA scorer-ranking reversal.

Section 5.3 reports that the raw slice ranking puts QNLI above the MS MARCO
cross-encoder and the adjusted ranking reverses it, and states that no interval is
available on a difference between two scorers' AUCs. This computes them.

Three margins, all on the one declared slice, one label vector, one population:

  raw       A_adv(s_qnli)   - A_adv(s_ce)      (QNLI leads raw)
  adjusted  A_adv(rep_ce)   - A_adv(rep_qnli)  (the cross-encoder leads adjusted)
  licensed  A_adv(rep_ce)   - A_adv(s_qnli)    (C2 refuses QNLI, so its raw stands)

Paired over the same bootstrap resample of questions, so the interval is on the
difference and not on two independent draws. Recipe is the paper's: GroupKFold(5) on
question id, ridge alpha=1, slice from the dense surface predictor's errors.
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

from feature_split_slice import load_wikiqa, overlap, slice_from

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
rng = np.random.default_rng(0)
NBOOT = 1000

q, ans, y, s_ce, qid, _ = load_wikiqa()
s_qnli = np.load(f"{CACHE}/wikiqa_qnli_s_8000.npy")
s_stsb = np.load(f"{CACHE}/wikiqa_stsb_s_8000.npy")

ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
la = np.array([len(x.split()) for x in ans], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([ov, la, lq, la - lq])
Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
phi = hstack([Xa, csr_matrix(dense)]).tocsr()
cv = GroupKFold(5)

_, adv = slice_from(dense, y, qid, cv, sparse=False)
print("n_adv =", int(adv.sum()), "queries touching slice =", len(np.unique(qid[adv])))

rep = {}
for tag, sc in [("ce", s_ce), ("qnli", s_qnli), ("stsb", s_stsb)]:
    g = cross_val_predict(Ridge(alpha=1.0), phi, sc, cv=cv, groups=qid)
    rep[tag] = sc - g

A = lambda u: float(roc_auc_score(y[adv], u[adv]))
point = {
    "raw":      {"ce": round(A(s_ce), 4), "qnli": round(A(s_qnli), 4), "stsb": round(A(s_stsb), 4)},
    "adjusted": {"ce": round(A(rep["ce"]), 4), "qnli": round(A(rep["qnli"]), 4),
                 "stsb": round(A(rep["stsb"]), 4)},
}
print(json.dumps(point, indent=1))

qs = np.unique(qid[adv])
by_q = {qq: np.where(adv & (qid == qq))[0] for qq in qs}

MARGINS = {
    "raw_qnli_minus_ce":        (s_qnli, s_ce),
    "adjusted_ce_minus_qnli":   (rep["ce"], rep["qnli"]),
    "licensed_repce_minus_rawqnli": (rep["ce"], s_qnli),
    "adjusted_ce_minus_stsb":   (rep["ce"], rep["stsb"]),
}
draws = {k: [] for k in MARGINS}
for _ in range(NBOOT):
    pick = rng.choice(qs, size=len(qs), replace=True)
    idx = np.concatenate([by_q[qq] for qq in pick])
    yy = y[idx]
    if yy.min() == yy.max():
        continue
    for k, (a, b) in MARGINS.items():
        draws[k].append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))

out = {"n_adv": int(adv.sum()), "n_queries": int(len(qs)), "point": point, "margins": {}}
for k, (a, b) in MARGINS.items():
    d = np.array(draws[k])
    out["margins"][k] = {
        "point": round(float(roc_auc_score(y[adv], a[adv]) - roc_auc_score(y[adv], b[adv])), 4),
        "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        "frac_above_zero": round(float((d > 0).mean()), 4),
        "nboot": int(len(d)),
    }
    print(k, out["margins"][k], flush=True)

with open(os.path.join(HERE, "reversal_interval_results.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwritten reversal_interval_results.json")
