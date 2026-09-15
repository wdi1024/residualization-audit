"""TriviaQA scorer replication (TMLR review, top experiment: generality across scorers + more
Figure-4 dose-response points). Repeats the TriviaQA answer-sentence-selection positive with THREE
different-family off-distribution cross-encoders. Everything but the scorer s is held fixed
(rows, phi, slice, repair), so the comparison is a clean scorer-only manipulation and each (scorer)
contributes a (C2 R^2 -> repair gain) point to the dose-response.
Run: python3 triviaqa_scorers.py
"""
import os, re, json, numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
CV = KFold(5, shuffle=True, random_state=0); rng = np.random.default_rng(0)

SCORERS = {
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "qnli-distilroberta-base": "cross-encoder/qnli-distilroberta-base",
    "stsb-roberta-base": "cross-encoder/stsb-roberta-base",
}

# rebuild the exact TriviaQA rows (deterministic, identical to triviaqa_qa_run.py)
d = load_dataset("lucadiliello/triviaqa", split="validation")
rows = []
for r in list(d)[:3000]:
    ctx = re.sub(r"\[[A-Z]+\]", " ", r["context"])
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if 25 < len(s.strip()) < 300]
    if len(sents) < 4: continue
    anss = [a.lower() for a in r["answers"]]
    pos = [s for s in sents if any(a in s.lower() for a in anss)]
    neg = [s for s in sents if not any(a in s.lower() for a in anss)]
    if not pos or len(neg) < 3: continue
    rows.append((r["question"], pos[0], 1))
    for s in neg[:3]: rows.append((r["question"], s, 0))
q = [a for a, _, _ in rows]; sent = [b for _, b, _ in rows]; y = np.array([c for _, _, c in rows])

def ov(a, b):
    A, B = set(a.lower().split()), set(b.lower().split()); return len(A & B) / max(1, len(A | B))
o = np.array([ov(a, b) for a, b in zip(q, sent)])[:, None]
ls = np.array([len(x.split()) for x in sent], float)[:, None]
lq = np.array([len(x.split()) for x in q], float)[:, None]
dense = np.hstack([o, ls, lq, ls - lq])
Xs = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(sent)
phi = hstack([Xs, csr_matrix(dense)]).tocsr()
art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        dense, y, cv=CV, method="predict_proba")[:, 1]
thr = np.quantile(art, 1 - y.mean()); adv = (art > thr) != (y == 1)
print(f"TriviaQA n={len(y)} C1={roc_auc_score(y, art):.3f} n_adv={int(adv.sum())} (fixed across scorers)")

def auc(m, x): return roc_auc_score(y[m], x[m])
out = []
for tag, model in SCORERS.items():
    cache = f"/tmp/triviaqa_{tag}.npy"
    if os.path.exists(cache):
        s = np.load(cache)
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model, max_length=256)
        s = np.asarray(ce.predict(list(zip(q, sent)), batch_size=64, show_progress_bar=False), float)
        np.save(cache, s)
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    r2 = 1 - float(((s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
    idx = np.where(adv)[0]; g = []
    for _ in range(1000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) > 1: g.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(g, [2.5, 97.5])
    row = {"scorer": tag, "raw_AUC": roc_auc_score(y, s), "C2_R2": r2,
           "A_adv_raw": auc(adv, s), "A_adv_rep": auc(adv, s_rep),
           "adv_gain": auc(adv, s_rep) - auc(adv, s), "adv_CI": [float(lo), float(hi)]}
    out.append(row)
    print(f"  {tag:26} rawAUC {row['raw_AUC']:.3f}  R2 {r2:.3f}  slice {row['A_adv_raw']:.3f}->{row['A_adv_rep']:.3f}"
          f"  gain {row['adv_gain']:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "notes", "triviaqa_scorers_2026-07-11.json"), "w"), indent=1)
print("\n=> TriviaQA positive replicates across scorers if all gains > 0; the (R2 -> gain) points extend Figure 4.")
