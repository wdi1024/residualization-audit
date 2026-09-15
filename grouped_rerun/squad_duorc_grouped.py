"""SQuAD/DuoRC held-out confirmatory rerun: row-KFold vs question-GroupKFold.

Row construction copied verbatim from heldout_{squad,duorc}_confirmatory.py
(deterministic, no RNG); scores from project caches. Alignment sanity-checked
by reproducing the paper-CV C1 before trusting the grouped delta.
"""
import json, os, re
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
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

def build_squad():
    d = load_dataset("squad", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1:
            continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0))
            qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000:
            break
    return rows[:8000], np.array(qid[:8000]), "squad_ce_s_8000.npy"

def build_duorc():
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["plot"])
        if not (3 <= len(sents) <= 25):
            continue
        if r["no_answer"] or not r["answers"]:
            continue
        ans = r["answers"][0]
        if len(ans.split()) > 12 or not ans.strip():
            continue
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1:
            continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0))
            qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000:
            break
    return rows[:8000], np.array(qid[:8000]), "duorc_ce_s_8000.npy"

def run(name, rows, qid, s_file):
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([r[2] for r in rows])
    s = np.load(os.path.join(CACHE, s_file))
    assert len(s) == len(y), f"{name}: cache mismatch {len(s)} vs {len(y)}"
    print(f"\n=== {name}: n={len(y)} pos={y.mean():.3f} questions={qid.max()+1}")

    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()

    res = {}
    for tag, cv, grp in [("paper", KFold(5, shuffle=True, random_state=0), None),
                         ("grouped", GroupKFold(5), qid)]:
        kw = dict(cv=cv) if grp is None else dict(cv=cv, groups=grp)
        art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                dense, y, method="predict_proba", **kw)[:, 1]
        c1 = roc_auc_score(y, art)
        thr = np.quantile(art, 1 - y.mean())
        adv = (art > thr) != (y == 1)
        pred = cross_val_predict(Ridge(alpha=1.0), phi, s, **kw)
        s_rep = s - pred
        r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
        out = {"C1": round(c1, 4), "C2_R2": round(r2, 4), "n_adv": int(adv.sum()),
               "adv_gain": round(roc_auc_score(y[adv], s_rep[adv]) - roc_auc_score(y[adv], s[adv]), 4),
               "full_gain": round(roc_auc_score(y, s_rep) - roc_auc_score(y, s), 4)}
        res[tag] = out
        print(f"[{tag}] {json.dumps(out)}")
    delta = {k: round(res["grouped"][k] - res["paper"][k], 4)
             for k in ("C1", "C2_R2", "adv_gain", "full_gain")}
    print("[delta]", json.dumps(delta))
    return {"paper": res["paper"], "grouped": res["grouped"], "delta": delta}

results = {}
results["squad"] = run("SQuAD", *build_squad())
results["duorc"] = run("DuoRC", *build_duorc())
json.dump(results, open(os.path.join(HERE, "squad_duorc_grouped_results.json"), "w"), indent=1)
print("\nwrote squad_duorc_grouped_results.json")
