"""Surgical 2nd-positive attempt: SICK (off-distribution for distilbert-MNLI).

Same recipe as nli_positive_run.py (SNLI positive): model trained on MNLI,
evaluated on a DIFFERENT dataset; phi = hypothesis-only TF-IDF; repair =
CV-Ridge residualization. SICK was skipped on 2026-07-09 (deprecated HF
loader); fetched here via the datasets-server mirror yangwang825/sick
(text1=sentence_A/premise, text2=sentence_B/hypothesis, label int).
Label semantics are recovered automatically from model agreement.

Gates from the second-positive hunt: C1 artifact-only AUC >= ~0.65,
C2 score genuinely corrupted (off-distribution). Reports both.
"""
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

torch.set_num_threads(6)
CV = KFold(n_splits=5, shuffle=True, random_state=0)
BASE = ("https://datasets-server.huggingface.co/rows?dataset=yangwang825%2Fsick"
        "&config=default&split=test")

rows = []
for off in range(0, 4906, 100):
    out = subprocess.run(["curl", "-sL", "--max-time", "60",
                          f"{BASE}&offset={off}&length=100"],
                         capture_output=True, check=True).stdout
    rows += [x["row"] for x in json.loads(out)["rows"]]
print("fetched", len(rows))

name = "typeform/distilbert-base-uncased-mnli"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForSequenceClassification.from_pretrained(name).eval()
ent_idx = [i for i, l in m.config.id2label.items() if "entail" in l.lower()][0]


def pent(p, h):
    with torch.no_grad():
        lo = m(**tok(p, h, return_tensors="pt", truncation=True,
                     max_length=256)).logits[0]
        return torch.softmax(lo, -1)[ent_idx].item()


# --- label semantics recovery on a probe sample: which int is entailment?
by_label = {}
for r in rows[:600]:
    by_label.setdefault(r["label"], []).append(r)
probe_means = {}
for lab, rs in sorted(by_label.items()):
    ss = [pent(r["text1"], r["text2"]) for r in rs[:40]]
    probe_means[lab] = float(np.mean(ss))
    print(f"label {lab}: n_probe={min(len(rs),40)} mean_p_ent={probe_means[lab]:.3f}")
ent_lab = max(probe_means, key=probe_means.get)
con_lab = min(probe_means, key=probe_means.get)
print(f"recovered: entailment={ent_lab}, contradiction={con_lab}")

pairs = [(r["text1"], r["text2"], 1 if r["label"] == ent_lab else 0)
         for r in rows if r["label"] in (ent_lab, con_lab)][:2400]
prem = [a for a, _, _ in pairs]
hyp = [b for _, b, _ in pairs]
y = np.array([c for _, _, c in pairs])
print(f"binary subset n={len(y)} (pos rate {y.mean():.3f})")

cache = "/tmp/sick_s.npy"
if os.path.exists(cache):
    s = np.load(cache)
    print("loaded cached scores")
else:
    s = []
    for i, (p, h) in enumerate(zip(prem, hyp)):
        s.append(pent(p, h))
        if i % 400 == 0:
            print("scored", i, flush=True)
    s = np.array(s)
    np.save(cache, s)

Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
hyp_prob = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y,
                             cv=CV, method="predict_proba")[:, 1]
s_rep = s - cross_val_predict(Ridge(alpha=1.0), Xh, s, cv=CV)
adv = (hyp_prob > 0.5) != (y == 1)
full = np.ones(len(y), bool)


def auc(mask, x):
    return roc_auc_score(y[mask], x[mask])


rng = np.random.default_rng(0)
gains = []
idx_adv = np.where(adv)[0]
for _ in range(1000):
    b = rng.choice(idx_adv, len(idx_adv), replace=True)
    if len(set(y[b])) < 2:
        continue
    gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
lo, hi = np.percentile(gains, [2.5, 97.5])

print(f"\n=== SICK (distilbert-mnli off-dist, n={len(y)}) — repair ===")
print(f"C1 artifact strength (hyp-only AUC): {roc_auc_score(y, hyp_prob):.3f}  (gate ~0.65)")
print(f"C2 raw model AUC (corrupted if visibly imperfect): {auc(full, s):.3f}")
print(f"{'':10}{'AUC full':>10}{'AUC adv slice':>16}")
print(f"{'raw s':10}{auc(full, s):>10.3f}{auc(adv, s):>16.3f}   (n_adv={int(adv.sum())})")
print(f"{'repaired':10}{auc(full, s_rep):>10.3f}{auc(adv, s_rep):>16.3f}")
print(f"adv-slice gain: {auc(adv, s_rep) - auc(adv, s):+.3f}  [CI {lo:+.3f},{hi:+.3f}]")
