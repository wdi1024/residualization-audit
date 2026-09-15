"""Round-8 review: break the y double-use with a second construct measurement.

(a) Synthetic two-reading control: c latent; y1, y2 are two independent noisy
    readings of c. Slice definition and C2 use y1 ONLY; validation AUC uses
    y2 ONLY. If the slice gain were an artifact of conditioning on the same
    label used for evaluation, it should vanish here; if it reflects the
    score's artifact-attributable error, it should survive (below the
    collinearity crossing) and still refuse above it.

(b) SNLI annotator-split: the original SNLI dev release carries five
    annotator labels per pair. y_A = majority of annotators {1,2,3} defines
    the artifact model, slice, and C2; y_B = majority of annotators {4,5}
    (disjoint humans) is used ONLY for validation. This is a real-data
    version of the same separation on the paper's own SNLI setting.

Writes ../notes/dual_measurement_check.json.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)
out = {}

# ---------------- (a) synthetic two-reading ----------------
def synth(rho, n=4000, noise=0.15):
    c = rng.normal(size=n)
    a = rho * c + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    phi = np.column_stack([a + 0.3 * rng.normal(size=n) for _ in range(6)])
    s = c + 1.2 * a + 0.5 * rng.normal(size=n)
    def reading():
        y = (c > 0).astype(int)
        flip = rng.random(n) < noise
        return np.where(flip, 1 - y, y)
    y1, y2 = reading(), reading()
    art = cross_val_predict(LogisticRegression(max_iter=1000), phi, y1,
                            cv=CV, method="predict_proba")[:, 1]
    adv = (art > 0.5) != (y1 == 1)          # slice from y1 only
    srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    g1 = roc_auc_score(y1[adv], srep[adv]) - roc_auc_score(y1[adv], s[adv])
    g2 = roc_auc_score(y2[adv], srep[adv]) - roc_auc_score(y2[adv], s[adv])
    return {"rho": rho, "n_adv": int(adv.sum()),
            "gain_on_y1_(circular)": float(g1),
            "gain_on_y2_(independent_reading)": float(g2)}

out["synthetic_two_reading"] = [synth(r) for r in (0.0, 0.2, 0.4, 0.6, 0.8)]

# ---------------- (b) SNLI annotator split ----------------
from datasets import load_dataset
d = load_dataset("stanfordnlp/snli", split="validation")
rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
        for r in d if r["label"] in (0, 2)][:2400]
hyp = [b for _, b, _ in rows]
s = np.load(os.path.join(HERE, "..", "cache", "snli_s_2400.npy"))

ann = {}
for line in open("/tmp/snli_1.0/snli_1.0_dev.jsonl"):
    r = json.loads(line)
    labs = [l for l in r["annotator_labels"] if l]
    ann[(r["sentence1"], r["sentence2"])] = labs

def maj(labels, idx):
    sel = [labels[i] for i in idx if i < len(labels)]
    e = sel.count("entailment"); c = sel.count("contradiction")
    if e > c:
        return 1
    if c > e:
        return 0
    return None

yA, yB, keep = [], [], []
for i, (p, h, _) in enumerate(rows):
    labs = ann.get((p, h))
    if not labs or len(labs) < 5:
        keep.append(False); yA.append(0); yB.append(0); continue
    a_ = maj(labs, [0, 1, 2]); b_ = maj(labs, [3, 4])
    if a_ is None or b_ is None:
        keep.append(False); yA.append(0); yB.append(0); continue
    keep.append(True); yA.append(a_); yB.append(b_)
keep = np.array(keep); yA = np.array(yA); yB = np.array(yB)
Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
k = np.where(keep)[0]
XhK, sK, yAK, yBK = Xh[k], s[k], yA[k], yB[k]
agree = float((yAK == yBK).mean())

art = cross_val_predict(LogisticRegression(max_iter=1000), XhK, yAK,
                        cv=CV, method="predict_proba")[:, 1]
adv = (art > 0.5) != (yAK == 1)             # slice from annotators {1,2,3}
srep = sK - cross_val_predict(Ridge(alpha=1.0), XhK, sK, cv=CV)
gA = roc_auc_score(yAK[adv], srep[adv]) - roc_auc_score(yAK[adv], sK[adv])
gB = roc_auc_score(yBK[adv], srep[adv]) - roc_auc_score(yBK[adv], sK[adv])
out["snli_annotator_split"] = {
    "n_kept": int(keep.sum()), "yA_yB_agreement": agree,
    "n_adv": int(adv.sum()),
    "slice_AUC_yA": [float(roc_auc_score(yAK[adv], sK[adv])),
                     float(roc_auc_score(yAK[adv], srep[adv]))],
    "slice_AUC_yB_independent": [float(roc_auc_score(yBK[adv], sK[adv])),
                                 float(roc_auc_score(yBK[adv], srep[adv]))],
    "gain_yA_(shares_slice_label)": float(gA),
    "gain_yB_(disjoint_annotators)": float(gB)}

json.dump(out, open(os.path.join(HERE, "..", "notes",
                                 "dual_measurement_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
