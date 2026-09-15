"""Score-scale sensitivity of residualization (review must-fix #7).

AUC of the raw score is invariant to monotone transforms, but s - g(phi) is not.
For each QA dataset (question-GroupKFold everywhere, i.e. the corrected regime),
residualize four representations of the same score and compare slice/full gains:
  logit  — cached cross-encoder output (paper's scale)
  prob   — sigmoid(logit)
  rankg  — rank -> Gaussian (scale-free)
  winz   — winsorized (1%/99%) z-score
Slice definition does not involve s, so it is identical across transforms.
"""
import json, os, re
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")

def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]

# ---------------- dataset loaders (identical row construction) ----------------
def load_wikiqa():
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    q = [a for a,_,_ in rows]; ans = [b for _,b,_ in rows]
    y = np.array([c for _,_,c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"), np.array([uniq[qq] for qq in q])

def load_asnq():
    rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
    q = [a for a,_,_ in rows]; ans = [b for _,b,_ in rows]
    y = np.array([c for _,_,c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/asnq_ce_s_8000.npy"), np.array([uniq[qq] for qq in q])

def load_trivia():
    rows = np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/triviaqa_ce_s.npy"), np.array([uniq[qq] for qq in q])

def load_squad():
    d = load_dataset("squad", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25): continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1: continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0)); qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000: break
    rows = rows[:8000]
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows]), np.load(f"{CACHE}/squad_ce_s_8000.npy"),
            np.array(qid[:8000]))

def load_duorc():
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["plot"])
        if not (3 <= len(sents) <= 25): continue
        if r["no_answer"] or not r["answers"]: continue
        ans = r["answers"][0]
        if len(ans.split()) > 12 or not ans.strip(): continue
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1: continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0)); qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000: break
    rows = rows[:8000]
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows]), np.load(f"{CACHE}/duorc_ce_s_8000.npy"),
            np.array(qid[:8000]))

# ---------------- transforms ----------------
def transforms(s):
    n = len(s)
    ranks = np.argsort(np.argsort(s)) + 0.5
    lo, hi = np.percentile(s, [1, 99])
    w = np.clip(s, lo, hi)
    return {
        "logit": s.astype(float),
        "prob": expit(s),
        "rankg": norm.ppf(ranks / n),
        "winz": (w - w.mean()) / w.std(),
    }

def run(name, q, ans, y, s, groups):
    print(f"\n=== {name} (question-GroupKFold)")
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    cv = GroupKFold(5)

    art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                            dense, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    thr = np.quantile(art, 1 - y.mean())
    adv = (art > thr) != (y == 1)

    out = {}
    for tag, st in transforms(s).items():
        pred = cross_val_predict(Ridge(alpha=1.0), phi, st, cv=cv, groups=groups)
        srep = st - pred
        out[tag] = {
            "adv_gain": round(roc_auc_score(y[adv], srep[adv]) - roc_auc_score(y[adv], st[adv]), 4),
            "full_gain": round(roc_auc_score(y, srep) - roc_auc_score(y, st), 4),
            "A_adv_rep": round(roc_auc_score(y[adv], srep[adv]), 4),
        }
        print(f"  [{tag:5s}] {json.dumps(out[tag])}")
    return out

results = {}
for name, loader in [("wikiqa", load_wikiqa), ("asnq", load_asnq),
                     ("triviaqa", load_trivia), ("squad", load_squad),
                     ("duorc", load_duorc)]:
    results[name] = run(name, *loader())
json.dump(results, open(os.path.join(HERE, "scale_sensitivity_results.json"), "w"), indent=1)
print("\nwrote scale_sensitivity_results.json")
