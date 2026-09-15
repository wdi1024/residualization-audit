"""Non-NLI positive hunt, round 2 (2026-07-10): C1 screen BEFORE neural compute.

Candidates chosen for documented LINEAR artifacts outside NLI:
  1. WikiQA answer selection  - answer-only lexical + overlap/length artifact
     (word-matching baselines are historically competitive on WikiQA)
  2. SHP human preference     - length bias, THE canonical reward-model artifact
  3. hh-rlhf preference       - same length screen, shorter responses
  4. HateCheck                - identity-term artifact (risk: templated design
     may have removed it, ANLI-style)
  5. BoolQ                    - question-only artifact (documented ~0.64 baseline)

Each prints artifact-only AUC (gate ~0.65). Only winners get model scoring.
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
import re

CV = KFold(5, shuffle=True, random_state=0)


def cv_auc(X, y, scale=False):
    clf = LogisticRegression(max_iter=1000)
    if scale:
        clf = make_pipeline(StandardScaler(with_mean=False), clf)
    p = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def wikiqa():
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    print(f"WikiQA n={len(y)} pos_rate={y.mean():.3f}")
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    print(f"  answer-only TF-IDF AUC      = {cv_auc(Xa, y):.3f}")
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    print(f"  overlap+lengths AUC         = {cv_auc(dense, y, scale=True):.3f}")
    Xfull = hstack([Xa, csr_matrix(dense)]).tocsr()
    print(f"  answer-TFIDF+dense AUC      = {cv_auc(Xfull, y, scale=True):.3f}")


def shp():
    d = load_dataset("stanfordnlp/SHP", split="test")
    rows = [(r["human_ref_A"], r["human_ref_B"], r["labels"]) for r in d][:4000]
    A = [a for a, _, _ in rows]
    B = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])   # 1 = A preferred
    print(f"SHP n={len(y)} A-preferred={y.mean():.3f}")
    la = np.array([len(x.split()) for x in A], float)[:, None]
    lb = np.array([len(x.split()) for x in B], float)[:, None]
    F = np.hstack([la, lb, la - lb, np.log1p(la) - np.log1p(lb)])
    print(f"  length-only AUC             = {cv_auc(F, y, scale=True):.3f}")


def hh():
    d = load_dataset("Anthropic/hh-rlhf", split="test")
    rows = [(r["chosen"], r["rejected"]) for r in d][:4000]
    # pair -> (A,B,y) with random-ish deterministic flip so y isn't constant
    A, B, y = [], [], []
    for i, (c, rj) in enumerate(rows):
        if i % 2 == 0:
            A.append(c); B.append(rj); y.append(1)
        else:
            A.append(rj); B.append(c); y.append(0)
    y = np.array(y)
    la = np.array([len(x.split()) for x in A], float)[:, None]
    lb = np.array([len(x.split()) for x in B], float)[:, None]
    F = np.hstack([la, lb, la - lb, np.log1p(la) - np.log1p(lb)])
    print(f"hh-rlhf n={len(y)}")
    print(f"  length-only AUC             = {cv_auc(F, y, scale=True):.3f}")


IDENT = ["black", "white", "muslim", "jewish", "christian", "gay", "lesbian",
         "homosexual", "trans", "transgender", "women", "woman", "female",
         "men", "man", "male", "immigrant", "mexican", "islam", "muslims",
         "jews", "gays", "queer", "asian", "african", "catholic", "atheist",
         "feminist", "disabled", "disability"]


def hatecheck():
    d = load_dataset("Paul/hatecheck", split="test")
    rows = [(r["test_case"], 1 if r["label_gold"] == "hateful" else 0) for r in d]
    texts = [t for t, _ in rows]
    y = np.array([c for _, c in rows])
    print(f"HateCheck n={len(y)} hateful={y.mean():.3f}")

    def ic(t):
        tl = t.lower()
        return [len(re.findall(r"\b" + re.escape(w) + r"\b", tl)) for w in IDENT]
    F = np.hstack([np.array([ic(t) for t in texts], float),
                   np.array([len(t.split()) for t in texts], float)[:, None]])
    print(f"  identity+length AUC         = {cv_auc(F, y, scale=True):.3f}")


def boolq():
    d = load_dataset("google/boolq", split="train")
    rows = [(r["question"], r["passage"], 1 if r["answer"] else 0) for r in d][:6000]
    q = [a for a, _, _ in rows]
    y = np.array([c for _, _, c in rows])
    print(f"BoolQ n={len(y)} yes={y.mean():.3f}")
    Xq = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(q)
    print(f"  question-only TF-IDF AUC    = {cv_auc(Xq, y):.3f}")


for fn in (wikiqa, shp, hh, hatecheck, boolq):
    try:
        fn()
    except Exception as e:
        print(f"{fn.__name__} FAILED: {str(e)[:120]}")
