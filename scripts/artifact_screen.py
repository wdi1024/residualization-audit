"""Screen candidate testbeds by PARTIAL-INPUT artifact strength BEFORE spending neural
scoring compute. Empirical criterion (from HANS + toxicity failures): the repair positive
needs an artifact-only AUC >= ~0.65 (a strong, LINEAR-recoverable artifact). Score each
candidate's partial-input -> label AUC with cross-fit TF-IDF+LogReg (no neural model, cheap).
Pick the strongest NON-SNLI domain, then run the full repair only on the winner.
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
CV = KFold(5, shuffle=True, random_state=0)

def artifact_auc(texts, y):
    y = np.asarray(y)
    X = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(texts)
    p = cross_val_predict(make_pipeline(TfidfVectorizer, None) if False else
                          LogisticRegression(max_iter=1000), X, y, cv=CV, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(y)

def try_cand(name, loader):
    try:
        texts, y, desc = loader()
        a, n = artifact_auc(texts, y)
        print(f"{name:28} artifact-only AUC = {a:.3f}   (n={n})   [{desc}]")
    except Exception as e:
        print(f"{name:28} FAIL: {str(e)[:80]}")

# MNLI hypothesis-only (NLI, sanity — expect ~0.67)
def mnli():
    d = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    rows = [(r["hypothesis"], 1 if r["label"] == 0 else 0) for r in d if r["label"] in (0, 2)][:4000]
    return [h for h, _ in rows], [c for _, c in rows], "hyp-only, entail-vs-contra"

# FEVER claim-only (fact verification — different domain, documented claim-only bias)
def fever():
    d = load_dataset("fever/fever", "v1.0", split="train", streaming=True)
    texts, y = [], []
    for r in d:
        lab = r["label"]
        if lab == "SUPPORTS": y.append(1)
        elif lab == "REFUTES": y.append(0)
        else: continue
        texts.append(r["claim"])
        if len(y) >= 4000: break
    return texts, y, "claim-only, SUPPORTS-vs-REFUTES"

# PAWS (paraphrase adversarial — lexical overlap artifact)
def paws():
    d = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    rows = [(r["sentence1"] + " [SEP] " + r["sentence2"], r["label"]) for r in d][:4000]
    return [t for t, _ in rows], [c for _, c in rows], "pair text, paraphrase label"

# QQP question2-only (paraphrase — partial-input artifact)
def qqp():
    d = load_dataset("nyu-mll/glue", "qqp", split="train")
    rows = [(r["question2"], r["label"]) for r in d][:4000]
    return [t for t, _ in rows], [c for _, c in rows], "q2-only, duplicate label"

# RTE hypothesis-only (small NLI)
def rte():
    d = load_dataset("nyu-mll/glue", "rte", split="train")
    rows = [(r["sentence2"], 1 - r["label"]) for r in d]   # 0=entail
    return [t for t, _ in rows], [c for _, c in rows], "hyp-only, entailment"

for nm, ld in [("MNLI(hyp-only)", mnli), ("FEVER(claim-only)", fever),
               ("PAWS(pair)", paws), ("QQP(q2-only)", qqp), ("RTE(hyp-only)", rte)]:
    try_cand(nm, ld)
print("\npick: strongest artifact-only AUC >= ~0.65 in a domain != SNLI, then full repair run.")
