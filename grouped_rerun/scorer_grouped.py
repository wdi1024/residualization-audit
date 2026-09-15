"""Grouped-CV analysis of the rescored scorer replications (ASNQ x3, TriviaQA x2).

For each scorer: row-KFold reproduction first (sanity against the 2026-07-11
records), then question-GroupKFold values: R2_phi_to_s, Delta_adv, adv-slice
gain with query-clustered bootstrap CI, R_lin before/after.
"""
import json, os
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
rng = np.random.default_rng(0)

def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

def load_asnq():
    rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
    return ([a for a, _, _ in rows], [b for _, b, _ in rows],
            np.array([c for _, _, c in rows]))

def load_trivia():
    rows = np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([int(r[2]) for r in rows]))

def q_boot_gain(y, raw, rep, mask, qid, nboot=1000):
    qs = np.unique(qid[mask])
    by_q = {qq: np.where(mask & (qid == qq))[0] for qq in qs}
    gains = []
    for _ in range(nboot):
        pick = rng.choice(qs, size=len(qs), replace=True)
        idx = np.concatenate([by_q[qq] for qq in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        gains.append(roc_auc_score(yy, rep[idx]) - roc_auc_score(yy, raw[idx]))
    return [round(float(np.percentile(gains, 2.5)), 4), round(float(np.percentile(gains, 97.5)), 4)]

def analyze(name, q, ans, y, score_files):
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    qid = np.array([uniq[qq] for qq in q])
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    out = {}
    for tag, cv, grp in [("row", KFold(5, shuffle=True, random_state=0), None),
                         ("grouped", GroupKFold(5), qid)]:
        kw = dict(cv=cv) if grp is None else dict(cv=cv, groups=grp)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        art = cross_val_predict(pipe, dense, y, method="predict_proba", **kw)[:, 1]
        chat = cross_val_predict(pipe, dense, y, method="decision_function", **kw)
        thr = np.quantile(art, 1 - y.mean())
        adv = (art > thr) != (y == 1)
        for m, sf in score_files.items():
            s = np.load(os.path.join(CACHE, sf)).astype(float)
            pred = cross_val_predict(Ridge(alpha=1.0), phi, s, **kw)
            srep = s - pred
            r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
            res = {
                "R2": round(r2, 4),
                "delta_adv": round(float(roc_auc_score(y, s) - roc_auc_score(y[adv], s[adv])), 4),
                "adv": [round(float(roc_auc_score(y[adv], s[adv])), 4),
                        round(float(roc_auc_score(y[adv], srep[adv])), 4)],
                "gain": round(float(roc_auc_score(y[adv], srep[adv]) - roc_auc_score(y[adv], s[adv])), 4),
                "gain_ci": q_boot_gain(y, s, srep, adv, qid),
                "Rlin": [round(abs(float(np.corrcoef(s, chat)[0, 1])), 4),
                         round(abs(float(np.corrcoef(srep, chat)[0, 1])), 4)],
            }
            out.setdefault(tag, {})[m] = res
            print(f"[{name}/{tag}/{m}] {json.dumps(res)}", flush=True)
    return out

results = {}
q, ans, y = load_asnq()
results["asnq"] = analyze("asnq", q, ans, y, {
    "stsb": "asnq_stsb_s_8000.npy", "qnli": "asnq_qnli_s_8000.npy",
    "electra": "asnq_electra_s_8000.npy"})
q, ans, y = load_trivia()
results["triviaqa"] = analyze("triviaqa", q, ans, y, {
    "stsb": "triviaqa_stsb_s.npy", "qnli": "triviaqa_qnli_s.npy"})
json.dump(results, open(os.path.join(HERE, "scorer_grouped_results.json"), "w"), indent=1)
print("wrote scorer_grouped_results.json", flush=True)
