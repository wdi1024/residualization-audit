"""Second non-NLI positive for generality (TMLR review): TriviaQA answer-SENTENCE selection, a QA
domain distinct from WikiQA. Same recipe: an off-distribution MS-MARCO cross-encoder scores
(question, candidate-sentence) relevance; construct y = does the sentence contain a gold answer;
phi = surface overlap/length + sentence TF-IDF. C1 (dense-overlap AUC) screened at 0.675 (>=0.65).
If repair raises construct alignment on the artifact-adversarial slice, TriviaQA is a second,
non-WikiQA QA positive. Run: python3 triviaqa_qa_run.py
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
print(f"TriviaQA-sentsel n={len(y)} pos_rate={y.mean():.3f}")

cache = "/tmp/triviaqa_ce_s.npy"
if os.path.exists(cache):
    s = np.load(cache); print("loaded cached scores")
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    s = np.asarray(ce.predict(list(zip(q, sent)), batch_size=64, show_progress_bar=False), float)
    np.save(cache, s)
print(f"raw AUC(y,s) = {roc_auc_score(y, s):.3f}  (off-distribution scorer)")

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
c1 = roc_auc_score(y, art)
thr = np.quantile(art, 1 - y.mean()); adv = (art > thr) != (y == 1); full = np.ones(len(y), bool)
s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
def auc(m, x): return roc_auc_score(y[m], x[m])
pred = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
z = (art > thr).astype(int)
def floor(x):
    p = cross_val_predict(LogisticRegression(max_iter=1000), x.reshape(-1, 1), z, cv=CV, method="predict_proba")[:, 1]
    return roc_auc_score(z, p)
idx = np.where(adv)[0]; g = []
for _ in range(1000):
    b = rng.choice(idx, len(idx), replace=True)
    if len(set(y[b])) > 1: g.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
lo, hi = np.percentile(g, [2.5, 97.5])

out = {"dataset": "TriviaQA-sentsel", "scorer": "ms-marco-MiniLM-L-6-v2", "n": len(y),
       "C1_dense": c1, "C2_R2_phi_to_s": r2, "C2_Delta_adv": auc(full, s) - auc(adv, s),
       "A_full_raw": auc(full, s), "A_full_rep": auc(full, s_rep),
       "A_adv_raw": auc(adv, s), "A_adv_rep": auc(adv, s_rep), "n_adv": int(adv.sum()),
       "adv_gain": auc(adv, s_rep) - auc(adv, s), "adv_CI": [float(lo), float(hi)],
       "floor_raw": floor(s), "floor_rep": floor(s_rep)}
print(json.dumps(out, indent=1))
print(f"\n=> TriviaQA adversarial-slice repair: {out['A_adv_raw']:.3f} -> {out['A_adv_rep']:.3f} "
      f"(gain {out['adv_gain']:+.3f} CI[{lo:+.3f},{hi:+.3f}]). A 2nd non-NLI QA positive if gain>0 & CI excludes 0.")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "notes", "triviaqa_2026-07-11.json"), "w"), indent=1)
