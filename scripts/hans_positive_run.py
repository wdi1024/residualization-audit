"""HANS positive-case testbed for proxy repair.

HANS (McCoy et al. 2019) is the canonical NLI artifact-adversarial set: every pair has
HIGH premise-hypothesis surface overlap, but half are non-entailment. A model that leans
on the overlap/hypothesis-surface artifact predicts "entailment" spuriously and fails the
non-entailment half. This is a textbook (score, construct, artifact) triple with an
INDEPENDENT construct label (gold entail vs non-entail), which RewardBench lacked.

score s   = model P(entailment)
construct y = 1[gold_label == entailment]      (independent of surface artifact by design)
artifact phi = overlap ratio + lengths + hypothesis-only TF-IDF (the surface cues)
repair    = cross-fit Ridge residual of s on phi (closes linear artifact channel)

Claim: residualizing s on phi RAISES AUC(y, s) (strips surface-cue-driven false entailment)
and closes the artifact floor -> valid positive repair on real data, canonical benchmark.

Run: python3 hans_positive_run.py [n_subsample] [model_id]
"""
import sys, numpy as np, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
MID = sys.argv[2] if len(sys.argv) > 2 else "typeform/distilbert-base-uncased-mnli"

# ---------- load HANS ----------
rows = [l.rstrip("\n").split("\t") for l in open("/tmp/hans_eval.txt")]
hdr = rows[0]; ix = {c: i for i, c in enumerate(hdr)}
data = rows[1:]
prem = [r[ix["sentence1"]] for r in data]
hyp  = [r[ix["sentence2"]] for r in data]
gold = [r[ix["gold_label"]] for r in data]
heur = [r[ix["heuristic"]] for r in data]
y_all = np.array([1 if g == "entailment" else 0 for g in gold])

# stratified subsample by (heuristic, gold) to keep the adversarial balance
rng = np.random.default_rng(0)
idx = np.arange(len(data))
if N < len(data):
    keys = np.array([f"{h}|{g}" for h, g in zip(heur, gold)])
    per = {}
    for k in np.unique(keys):
        pool = idx[keys == k]; per[k] = rng.choice(pool, min(len(pool), N // len(np.unique(keys))), replace=False)
    idx = np.sort(np.concatenate(list(per.values())))
prem = [prem[i] for i in idx]; hyp = [hyp[i] for i in idx]
heur = [heur[i] for i in idx]; y = y_all[idx]
print(f"HANS n={len(y)}  entail={y.mean():.3f}  model={MID}")

# ---------- score s = model P(entailment) ----------
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID).eval()
ent_ix = next(i for i, l in model.config.id2label.items() if "entail" in l.lower())
print(f"entailment index = {ent_ix}  id2label={model.config.id2label}")

import os, hashlib
cache = f"/tmp/hans_s_{hashlib.md5((MID+str(N)).encode()).hexdigest()[:8]}.npy"
if os.path.exists(cache):
    s = np.load(cache); print(f"loaded cached scores {cache}")
else:
    s = np.zeros(len(y)); B = 32
    with torch.no_grad():
        for b in range(0, len(y), B):
            enc = tok(prem[b:b+B], hyp[b:b+B], truncation=True, padding=True,
                      max_length=128, return_tensors="pt")
            p = model(**enc).logits.softmax(-1)[:, ent_ix].numpy()
            s[b:b+B] = p
            if b % 1024 == 0: print(f"  scored {b}/{len(y)}", flush=True)
    np.save(cache, s)

# ---------- artifact phi = overlap + lengths + hypothesis-only TF-IDF ----------
def overlap_ratio(p, h):
    hs = h.lower().split(); ps = set(p.lower().split())
    return sum(w in ps for w in hs) / max(1, len(hs))
ov = np.array([overlap_ratio(p, h) for p, h in zip(prem, hyp)])
lp = np.array([len(p.split()) for p in prem]); lh = np.array([len(h.split()) for h in hyp])
dense = np.c_[ov, lp, lh, lp - lh]
tf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
Xh = tf.fit_transform(hyp)                      # hypothesis-only artifact
PHI_MODE = sys.argv[3] if len(sys.argv) > 3 else "dense"  # dense (documented HANS artifact) | tfidf | full
if PHI_MODE == "dense":   phi = csr_matrix(dense)
elif PHI_MODE == "tfidf": phi = Xh.tocsr()
else:                     phi = hstack([csr_matrix(dense), Xh]).tocsr()
print(f"[phi mode={PHI_MODE}  dims={phi.shape[1]}]")

# ---------- repair: cross-fit (standardized) Ridge residual ----------
CV = KFold(n_splits=5, shuffle=True, random_state=0)
def _reg(phi):
    sc = StandardScaler(with_mean=not hasattr(phi, "toarray"))  # sparse -> no centering
    return make_pipeline(sc, Ridge(alpha=1.0))
def repair(s, phi):
    return s - cross_val_predict(_reg(phi), phi, s, cv=CV)
def auc(y, x): return roc_auc_score(y, x)
def surface_R2(x, phi):   # certify: how much of the score is explained by surface artifact phi?
    pred = cross_val_predict(_reg(phi), phi, x, cv=CV)
    ss_res = float(((x - pred) ** 2).sum()); ss_tot = float(((x - x.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot

s_rep = repair(s, phi)

print("\n================ HANS proxy-repair ================")
print(f"{'':22}{'AUC(y=entail)':>14}{'surface-R2(phi->s)':>20}")
for tag, x in [("raw s=P(entail)", s), ("repaired (Ridge res)", s_rep)]:
    print(f"{tag:22}{auc(y, x):>14.3f}{surface_R2(x, phi):>20.3f}")

def boot_gap(y, a, b, R=2000):   # bootstrap CI on AUC(b)-AUC(a)
    n = len(y); g = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, n)
        if len(set(y[ii])) < 2: g[r] = np.nan; continue
        g[r] = auc(y[ii], b[ii]) - auc(y[ii], a[ii])
    g = g[~np.isnan(g)]
    return np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5)

m, lo, hi = boot_gap(y, s, s_rep)
print(f"\nglobal AUC gain (repaired - raw) = {m:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]")

# per-heuristic: AUC gain + surface-R2 of raw score (is the artifact channel open here?)
print("\nper-heuristic:   AUC raw->rep   gain[95%CI]        surfR2(phi->s_raw)   (n)")
for h in sorted(set(heur)):
    m2 = np.array([hh == h for hh in heur])
    if len(set(y[m2])) < 2: continue
    gm, gl, gh = boot_gap(y[m2], s[m2], s_rep[m2])
    r2 = surface_R2(s[m2], phi[m2])
    print(f"  {h:14}{auc(y[m2],s[m2]):.3f}->{auc(y[m2],s_rep[m2]):.3f}  {gm:+.3f}[{gl:+.3f},{gh:+.3f}]   {r2:>8.3f}   ({m2.sum()})")

# non-entailment recall proxy: accuracy at threshold on the misleading half
def acc_nonent(x):
    thr = np.median(x)
    pred_ent = (x > thr).astype(int)
    ne = y == 0
    return (pred_ent[ne] == 0).mean()
print(f"\nnon-entailment 'not-entail' rate @median:  raw {acc_nonent(s):.3f} -> repaired {acc_nonent(s_rep):.3f}")
print("expect: AUC(y) UP and artifact-floor -> ~0.5  => valid positive repair on canonical HANS")
