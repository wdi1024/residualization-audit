"""3rd-positive attempt: multiple-choice ending-only artifacts (HellaSwag, SWAG).

The 07-09 hunt named Story Cloze / SWAG ending-only as the remaining
high-probability positive family (documented strong partial-input artifact),
deferred as "more setup than value". With SICK secured, this is the optional
extra: flatten MC items to (context, ending, y) pairs; s = distilbert-MNLI
entailment prob (OFF-distribution -> C2 candidate by construction);
phi = ending-only TF-IDF. Same repair operator and gates as SNLI/SICK.
CV uses GroupKFold by item so same-item endings never cross folds.
"""
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

torch.set_num_threads(6)
N_ITEMS = 600

name = "typeform/distilbert-base-uncased-mnli"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForSequenceClassification.from_pretrained(name).eval()
ent_idx = [i for i, l in m.config.id2label.items() if "entail" in l.lower()][0]


def pent(p, h):
    with torch.no_grad():
        lo = m(**tok(p, h, return_tensors="pt", truncation=True,
                     max_length=256)).logits[0]
        return torch.softmax(lo, -1)[ent_idx].item()


def fetch(dataset, config, split, n_rows):
    rows = []
    for off in range(0, n_rows, 100):
        out = subprocess.run(
            ["curl", "-sL", "--max-time", "60",
             f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
             f"&config={config}&split={split}&offset={off}&length=100"],
            capture_output=True, check=True).stdout
        rows += [x["row"] for x in json.loads(out)["rows"]]
    return rows


def flatten_hellaswag(rows):
    for i, r in enumerate(rows):
        ctx = r["ctx"]
        for j, e in enumerate(r["endings"]):
            yield i, ctx, e, 1 if int(r["label"]) == j else 0


def flatten_swag(rows):
    for i, r in enumerate(rows):
        ctx = r["startphrase"]
        for j in range(4):
            yield i, ctx, r[f"ending{j}"], 1 if int(r["label"]) == j else 0


def run_one(tag, dataset, config, split, flat):
    rows = fetch(dataset, config, split, N_ITEMS)
    pairs = list(flat(rows))
    groups = np.array([g for g, _, _, _ in pairs])
    ctxs = [c for _, c, _, _ in pairs]
    ends = [e for _, _, e, _ in pairs]
    y = np.array([v for _, _, _, v in pairs])
    cache = f"/tmp/mc_{tag}_s.npy"
    if os.path.exists(cache):
        s = np.load(cache)
        print(tag, "loaded cached scores")
    else:
        s = []
        for i, (c, e) in enumerate(zip(ctxs, ends)):
            s.append(pent(c, e))
            if i % 400 == 0:
                print(tag, "scored", i, flush=True)
        s = np.array(s)
        np.save(cache, s)

    cv = GroupKFold(n_splits=5)
    Xe = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ends)
    end_prob = cross_val_predict(LogisticRegression(max_iter=1000), Xe, y,
                                 cv=cv, groups=groups,
                                 method="predict_proba")[:, 1]
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), Xe, s, cv=cv, groups=groups)
    adv = (end_prob > np.median(end_prob)) != (y == 1)
    full = np.ones(len(y), bool)

    def auc(mask, x):
        return roc_auc_score(y[mask], x[mask])

    rng = np.random.default_rng(0)
    idx_adv = np.where(adv)[0]
    gains = []
    for _ in range(1000):
        b = rng.choice(idx_adv, len(idx_adv), replace=True)
        if len(set(y[b])) < 2:
            continue
        gains.append(roc_auc_score(y[b], s_rep[b]) - roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(gains, [2.5, 97.5])

    print(f"\n=== {tag} (distilbert-mnli off-dist, {len(y)} pairs / "
          f"{len(rows)} items) ===")
    print(f"C1 artifact (ending-only AUC): {roc_auc_score(y, end_prob):.3f}  (gate ~0.65)")
    print(f"C2 raw model AUC: {auc(full, s):.3f}")
    print(f"{'':10}{'AUC full':>10}{'AUC adv slice':>16}")
    print(f"{'raw s':10}{auc(full, s):>10.3f}{auc(adv, s):>16.3f}   (n_adv={int(adv.sum())})")
    print(f"{'repaired':10}{auc(full, s_rep):>10.3f}{auc(adv, s_rep):>16.3f}")
    print(f"adv-slice gain: {auc(adv, s_rep) - auc(adv, s):+.3f}  [CI {lo:+.3f},{hi:+.3f}]")


run_one("hellaswag", "Rowan%2Fhellaswag", "default", "validation", flatten_hellaswag)
run_one("swag", "allenai%2Fswag", "regular", "validation", flatten_swag)
