"""Second positive-case testbed for proxy repair: TOXICITY with the identity-term artifact.

CivilComments (Borkan et al. 2019 / WILDS) is the textbook spurious-correlation benchmark:
toxicity classifiers over-flag comments that merely MENTION identity groups (gay, muslim,
black, ...) even when non-toxic. That identity-lexical channel is a documented, LINEAR
artifact, and there is an INDEPENDENT human construct label (annotator toxicity fraction).
Different domain from NLI -> tests cross-domain generality of the repair.

score s   = toxic-bert P(toxic)
construct y = 1[toxicity >= 0.5]                 (human annotation, independent of identity words)
artifact phi = identity-term counts + comment length   (the documented linear artifact)
repair    = cross-fit (shuffled) Ridge residual of s on phi

Claim: on the artifact-adversarial slice (identity terms present but NON-toxic, where the
classifier over-flags), residualizing s on the identity channel RAISES AUC(y, s).

Run: python3 toxicity_positive_run.py [n]
"""
import sys, os, re, numpy as np, torch
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
CV = KFold(n_splits=5, shuffle=True, random_state=0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

# documented CivilComments identity terms (the spurious lexical channel)
IDENT = ["black","white","muslim","jewish","christian","gay","lesbian","homosexual","trans",
         "transgender","women","woman","female","men","man","male","immigrant","mexican",
         "islam","muslims","jews","gays","queer","asian","african","catholic","atheist","feminist"]

# ---------- load CivilComments: balance toxic/non-toxic, ensure identity-bearing coverage ----------
d = load_dataset("google/civil_comments", split="train", streaming=True)
pos, neg = [], []
idre = re.compile(r"\b(" + "|".join(IDENT) + r")\b", re.I)
for r in d:
    t = r["text"]; tox = r["toxicity"]
    if tox is None or not t.strip(): continue
    lab = 1 if tox >= 0.5 else (0 if tox <= 0.1 else None)   # clear pos / clear neg
    if lab is None: continue
    (pos if lab == 1 else neg).append(t)
    if len(pos) >= N // 2 and len(neg) >= N // 2: break
texts = pos[:N//2] + neg[:N//2]
y = np.array([1]*len(pos[:N//2]) + [0]*len(neg[:N//2]))
print(f"CivilComments n={len(y)}  toxic={y.mean():.3f}  identity-bearing={np.mean([bool(idre.search(t)) for t in texts]):.3f}")

# ---------- score s = toxic-bert P(toxic) ----------
MID = "unitary/toxic-bert"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID).eval()
tox_ix = next(i for i, l in model.config.id2label.items() if l.lower() == "toxic")
print(f"toxic index={tox_ix}  id2label={model.config.id2label}")

cache = f"/tmp/tox_s_{len(y)}.npy"
if os.path.exists(cache):
    s = np.load(cache); print("loaded cached scores")
else:
    s = np.zeros(len(y)); B = 16
    with torch.no_grad():
        for b in range(0, len(y), B):
            enc = tok(texts[b:b+B], truncation=True, padding=True, max_length=128, return_tensors="pt")
            s[b:b+B] = model(**enc).logits.sigmoid()[:, tox_ix].numpy()   # multilabel -> sigmoid
            if b % 512 == 0: print(f"  scored {b}/{len(y)}", flush=True)
    np.save(cache, s)

# ---------- artifact phi = identity-term counts + length ----------
def ident_counts(t):
    tl = t.lower()
    return [len(re.findall(r"\b"+re.escape(w)+r"\b", tl)) for w in IDENT]
ic = np.array([ident_counts(t) for t in texts], float)
ln = np.array([len(t.split()) for t in texts], float)[:, None]
has_ident = (ic.sum(1) > 0)
phi = csr_matrix(np.hstack([ic, ln, ic.sum(1, keepdims=True)]))

# ---------- repair ----------
def _reg(): return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
s_rep = s - cross_val_predict(_reg(), phi, s, cv=CV)
def auc(mask, x): return roc_auc_score(y[mask], x[mask])
def surface_R2(x):
    pred = cross_val_predict(_reg(), phi, x, cv=CV)
    return 1 - float(((x-pred)**2).sum())/float(((x-x.mean())**2).sum())

# SNLI-recipe adversarial slice: where an IDENTITY-ONLY predictor is WRONG (artifact misleads).
ident_prob = cross_val_predict(make_pipeline(StandardScaler(with_mean=False),
                LogisticRegression(max_iter=1000)), phi, y, cv=CV, method="predict_proba")[:, 1]
print(f"identity-only AUC (artifact strength) = {roc_auc_score(y, ident_prob):.3f}")
adv = (ident_prob > 0.5) != (y == 1)
identm = has_ident                      # WILDS subpopulation where the identity channel can act
allm = np.ones(len(y), bool)
print(f"identity-bearing-only AUC: raw {auc(identm,s):.3f} -> repaired {auc(identm,s_rep):.3f} (n={identm.sum()})")

def boot_gap(mask, R=2000):
    y_, a, b = y[mask], s[mask], s_rep[mask]; n = mask.sum(); g = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, n)
        if len(set(y_[ii])) < 2: g[r] = np.nan; continue
        g[r] = roc_auc_score(y_[ii], b[ii]) - roc_auc_score(y_[ii], a[ii])
    g = g[~np.isnan(g)]; return np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5)

print(f"\n=== TOXICITY (CivilComments, toxic-bert, n={len(y)}) — proxy repair ===")
print(f"surface-R2(identity phi -> s) = {surface_R2(s):.3f}   (artifact channel strength)")
print(f"{'':10}{'AUC full':>10}{'AUC adversarial slice':>24}")
print(f"{'raw s':10}{auc(allm,s):>10.3f}{auc(adv,s):>24.3f}   (n_adv={adv.sum()})")
print(f"{'repaired':10}{auc(allm,s_rep):>10.3f}{auc(adv,s_rep):>24.3f}")
gm, gl, gh = boot_gap(adv)
print(f"adversarial-slice AUC gain = {gm:+.3f}  95%CI [{gl:+.3f}, {gh:+.3f}]")
gm2, gl2, gh2 = boot_gap(allm)
print(f"full-set        AUC gain = {gm2:+.3f}  95%CI [{gl2:+.3f}, {gh2:+.3f}]")
print("expect: adversarial slice UP (repair strips identity over-flag), full may fall = validity condition")
