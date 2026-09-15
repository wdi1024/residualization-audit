"""Surgical 2nd positive: replicate the SNLI recipe = model trained on dataset A applied to a
DIFFERENT dataset B (cross-dataset), so the hyp-only artifact genuinely corrupts the score.

score s = distilbert-MNLI P(entail) on eval dataset B (SICK or ANLI)
construct y = 1[entail] (binary entail vs contra), artifact φ = hypothesis-only TF-IDF
repair = cross-fit (shuffled) Ridge residual. Report artifact strength, adversarial-slice AUC.

Run: python3 cross_nli_run.py <sick|anli>
"""
import sys, os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6)
rng = np.random.default_rng(0); CV = KFold(5, shuffle=True, random_state=0)
WHICH = sys.argv[1] if len(sys.argv) > 1 else "sick"

def load_pairs(which):
    if which == "sick":
        d = load_dataset("RobZamp/sick", split="test")
        lab = {"ENTAILMENT": 1, "CONTRADICTION": 0}   # label_str: ENTAILMENT/NEUTRAL/CONTRADICTION
        rows = [(r["sentence_A"], r["sentence_B"], lab[r["label"]] if isinstance(r["label"], str) else
                 (1 if r["label"] == 0 else (0 if r["label"] == 2 else None))) for r in d]
    elif which == "mnli":  # eval MNLI matched (hyp-only artifact ~0.62) with a SNLI-trained model (off-dist)
        d = load_dataset("nyu-mll/multi_nli", split="validation_matched")
        rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else (0 if r["label"] == 2 else None)) for r in d]
    else:  # anli (adversarial NLI) test rounds combined
        d = load_dataset("facebook/anli", split="test_r3")
        rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else (0 if r["label"] == 2 else None)) for r in d]
    rows = [r for r in rows if r[2] is not None]
    return [a for a, _, _ in rows], [b for _, b, _ in rows], np.array([c for _, _, c in rows])

prem, hyp, y = load_pairs(WHICH)
if len(y) > 4200:   # cap for CPU scoring; stratify-ish by taking a shuffled slice
    sel = rng.permutation(len(y))[:4000]
    prem = [prem[i] for i in sel]; hyp = [hyp[i] for i in sel]; y = y[sel]
print(f"{WHICH}: n={len(y)}  entail={y.mean():.3f}")

# artifact strength screen (hyp-only)
Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
art = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y, cv=CV, method="predict_proba")[:, 1]
art_auc = roc_auc_score(y, art)
print(f"hyp-only artifact AUC = {art_auc:.3f}  (need >=~0.65 for a chance at the positive)")

MID = sys.argv[2] if len(sys.argv) > 2 else "typeform/distilbert-base-uncased-mnli"
tok = AutoTokenizer.from_pretrained(MID); model = AutoModelForSequenceClassification.from_pretrained(MID).eval()
cache = f"/tmp/cross_{WHICH}_{MID.split('/')[-1]}_probs.npy"   # store all 3 class probs
if os.path.exists(cache):
    P = np.load(cache); print("loaded cached probs")
else:
    P = np.zeros((len(y), model.config.num_labels)); B = 32
    with torch.no_grad():
        for b in range(0, len(y), B):
            enc = tok(prem[b:b+B], hyp[b:b+B], truncation=True, padding=True, max_length=128, return_tensors="pt")
            P[b:b+B] = model(**enc).logits.softmax(-1).numpy()
    np.save(cache, P)
# pick the entailment column = the class prob best separating gold-entailment (label-order agnostic)
aucs = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
ent_ix = int(np.argmax(aucs)); s = P[:, ent_ix]
print(f"model={MID}  per-class AUC(entail)={[f'{a:.3f}' for a in aucs]} -> ent_ix={ent_ix}")
print(f"raw model AUC(entail) = {roc_auc_score(y, s):.3f}  (want strong but NOT near-perfect: artifact corrupts it)")

def _reg(): return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
s_rep = s - cross_val_predict(_reg(), Xh, s, cv=CV)
adv = (art > 0.5) != (y == 1); allm = np.ones(len(y), bool)
def auc(m, x): return roc_auc_score(y[m], x[m])
def boot(m, R=2000):
    y_, a, b = y[m], s[m], s_rep[m]; n = int(m.sum()); g = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, n)
        g[r] = (roc_auc_score(y_[ii], b[ii]) - roc_auc_score(y_[ii], a[ii])) if len(set(y_[ii])) > 1 else np.nan
    g = g[~np.isnan(g)]; return np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5)

print(f"\n=== {WHICH} cross-dataset repair (distilbert-MNLI) ===")
print(f"{'':10}{'AUC full':>10}{'AUC adversarial slice':>24}")
print(f"{'raw s':10}{auc(allm,s):>10.3f}{auc(adv,s):>24.3f}  (n_adv={int(adv.sum())})")
print(f"{'repaired':10}{auc(allm,s_rep):>10.3f}{auc(adv,s_rep):>24.3f}")
gm, gl, gh = boot(adv); print(f"adversarial-slice gain = {gm:+.3f}  95%CI [{gl:+.3f},{gh:+.3f}]")
gm2, gl2, gh2 = boot(allm); print(f"full-set        gain = {gm2:+.3f}  95%CI [{gl2:+.3f},{gh2:+.3f}]")
print("positive = adversarial-slice gain > 0 with CI excluding 0")
