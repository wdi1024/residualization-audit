"""ASNQ / TriviaQA grouped-CV rerun (same recipe as wikiqa_grouped.py).

Rows come from project caches; group key = question string. SQuAD/DuoRC held-out
scripts rebuild rows from HF — handled in squad_duorc_grouped.py.
"""
import json, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")

def load_asnq():
    rows = json.load(open(os.path.join(CACHE, "asnq_rows_8000.json")))
    s = np.load(os.path.join(CACHE, "asnq_ce_s_8000.npy"))
    q = [a for a, _, _ in rows]; ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    return q, ans, y, s

def load_trivia():
    rows = np.load(os.path.join(CACHE, "triviaqa_rows.npy"), allow_pickle=True)
    s = np.load(os.path.join(CACHE, "triviaqa_ce_s.npy"))
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    return q, ans, y, s

def run_dataset(name, q, ans, y, s):
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    groups = np.array([uniq[qq] for qq in q])
    print(f"\n=== {name}: n={len(y)} pos={y.mean():.3f} questions={len(uniq)}")

    def overlap(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / max(1, len(A | B))
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()

    res = {}
    for tag, cv, grp in [("paper", KFold(5, shuffle=True, random_state=0), None),
                         ("grouped", GroupKFold(5), groups)]:
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
results["asnq"] = run_dataset("ASNQ", *load_asnq())
results["triviaqa"] = run_dataset("TriviaQA", *load_trivia())
json.dump(results, open(os.path.join(HERE, "qa_grouped_results.json"), "w"), indent=1)
print("\nwrote qa_grouped_results.json")
