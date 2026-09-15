"""Second-domain positive-case for proxy repair: QQP (duplicate-question detection).

Screened winner: q2-only artifact-only AUC = 0.700 (>= 0.65 bar), and a DIFFERENT domain
from NLI (paraphrase / duplicate detection). QQP has a documented partial-input annotation
artifact: some questions are "duplicate-prone" regardless of the paired question. A model that
leans on that q2-surface artifact mispredicts on the slice where the artifact is wrong.

score s   = QQP model P(duplicate)
construct y = QQP duplicate label
artifact phi = q2-only TF-IDF (the strong partial-input artifact) + overlap + lengths
repair    = cross-fit (shuffled) Ridge residual of s on phi

Claim: on the artifact-adversarial slice (q2-only predictor WRONG), residualizing s on the
q2-surface channel RAISES AUC(y, s).  Run: python3 qqp_positive_run.py [n] [model_id]
"""
import sys, os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
torch.set_num_threads(6)
rng = np.random.default_rng(0)
CV = KFold(5, shuffle=True, random_state=0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
MID = sys.argv[2] if len(sys.argv) > 2 else "textattack/bert-base-uncased-QQP"

# ---------- load QQP (balanced) ----------
d = load_dataset("nyu-mll/glue", "qqp", split="train")
pos = [(r["question1"], r["question2"]) for r in d if r["label"] == 1]
neg = [(r["question1"], r["question2"]) for r in d if r["label"] == 0]
k = N // 2
q1 = [a for a, _ in pos[:k]] + [a for a, _ in neg[:k]]
q2 = [b for _, b in pos[:k]] + [b for _, b in neg[:k]]
y = np.array([1]*len(pos[:k]) + [0]*len(neg[:k]))
print(f"QQP n={len(y)}  dup={y.mean():.3f}  model={MID}")

# ---------- score s = model P(duplicate) ----------
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID).eval()
id2 = {i: str(l).lower() for i, l in model.config.id2label.items()}
dup_ix = next((i for i, l in id2.items() if "dup" in l or l in ("label_1", "1", "equivalent")), 1)
print(f"duplicate index={dup_ix}  id2label={model.config.id2label}")

cache = f"/tmp/qqp_s_{len(y)}.npy"
if os.path.exists(cache):
    s = np.load(cache); print("loaded cached scores")
else:
    s = np.zeros(len(y)); B = 16
    with torch.no_grad():
        for b in range(0, len(y), B):
            enc = tok(q1[b:b+B], q2[b:b+B], truncation=True, padding=True, max_length=128, return_tensors="pt")
            s[b:b+B] = model(**enc).logits.softmax(-1)[:, dup_ix].numpy()
            if b % 512 == 0: print(f"  scored {b}/{len(y)}", flush=True)
    np.save(cache, s)

# ---------- artifact phi = q2-only TF-IDF + overlap + lengths ----------
def overlap(a, b):
    A, B_ = set(a.lower().split()), set(b.lower().split())
    return len(A & B_) / max(1, len(A | B_))
ov = np.array([overlap(a, b) for a, b in zip(q1, q2)])[:, None]
l1 = np.array([len(a.split()) for a in q1], float)[:, None]
l2 = np.array([len(b.split()) for b in q2], float)[:, None]
Xq2 = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(q2)   # q2-only artifact
phi = hstack([Xq2, csr_matrix(np.hstack([ov, l1, l2, l1 - l2]))]).tocsr()

# ---------- repair ----------
def _reg(): return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
s_rep = s - cross_val_predict(_reg(), phi, s, cv=CV)
def auc(mask, x): return roc_auc_score(y[mask], x[mask])
def surface_R2(x):
    pred = cross_val_predict(_reg(), phi, x, cv=CV)
    return 1 - float(((x-pred)**2).sum())/float(((x-x.mean())**2).sum())

# artifact-adversarial slice: q2-only predictor is WRONG
art_prob = cross_val_predict(make_pipeline(StandardScaler(with_mean=False),
              LogisticRegression(max_iter=1000)), Xq2, y, cv=CV, method="predict_proba")[:, 1]
print(f"q2-only artifact AUC = {roc_auc_score(y, art_prob):.3f}")
adv = (art_prob > 0.5) != (y == 1)
allm = np.ones(len(y), bool)

def boot_gap(mask, R=2000):
    y_, a, b = y[mask], s[mask], s_rep[mask]; n = int(mask.sum()); g = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, n)
        if len(set(y_[ii])) < 2: g[r] = np.nan; continue
        g[r] = roc_auc_score(y_[ii], b[ii]) - roc_auc_score(y_[ii], a[ii])
    g = g[~np.isnan(g)]; return np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5)

print(f"\n=== QQP (duplicate detection, {MID.split('/')[-1]}, n={len(y)}) — proxy repair ===")
print(f"surface-R2(q2 phi -> s) = {surface_R2(s):.3f}   (artifact channel strength in the score)")
print(f"{'':10}{'AUC full':>10}{'AUC adversarial slice':>24}")
print(f"{'raw s':10}{auc(allm,s):>10.3f}{auc(adv,s):>24.3f}   (n_adv={int(adv.sum())})")
print(f"{'repaired':10}{auc(allm,s_rep):>10.3f}{auc(adv,s_rep):>24.3f}")
gm, gl, gh = boot_gap(adv);  print(f"adversarial-slice AUC gain = {gm:+.3f}  95%CI [{gl:+.3f}, {gh:+.3f}]")
gm2, gl2, gh2 = boot_gap(allm); print(f"full-set        AUC gain = {gm2:+.3f}  95%CI [{gl2:+.3f}, {gh2:+.3f}]")
print("expect: adversarial slice UP (repair strips q2-surface over-reliance), full may fall = validity condition")
