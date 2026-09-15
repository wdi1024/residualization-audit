"""Round-2 review experiments (all offline given cached scores):

E1  certificate vs the LINEAR channel index (logistic decision_function,
    exactly w^T phi) alongside the probability version, per positive setting.
E2  query-clustered bootstrap CI for the adversarial-slice gain, plus
    per-query AUC and MRR (raw vs repaired), for the QA settings.
E3  feature-split ablation: repair with full phi vs answer-only phi vs
    overlap-only phi, same fixed slice, per QA setting.
E4  nested split-half slice protocol (slice from half A, repair and evaluate
    on half B; symmetric) extended to ASNQ and TriviaQA.

Writes ../notes/review_r2_experiments.json.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def build(rows):
    q = [r[0] for r in rows]
    ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi_full = hstack([Xa, csr_matrix(dense)]).tocsr()
    phi_ans = hstack([Xa, csr_matrix(np.hstack([la]))]).tocsr()   # question-blind
    phi_ov = csr_matrix(np.hstack([ov, la - lq]))                  # overlap-led
    qid = np.array([hash(x) for x in q])
    return q, ans, y, dense, phi_full, phi_ans, phi_ov, qid


def repair(phi, s):
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    return s - g


def slice_from(dense, y):
    art = cross_val_predict(make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=1000)),
                            dense, y, cv=CV, method="predict_proba")[:, 1]
    thr = np.quantile(art, 1 - y.mean())
    adv = (art > thr) != (y == 1)
    return art, adv


def lin_index(dense, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_predict(m, dense, y, cv=CV, method="decision_function")


def per_query_metrics(y, s, qid):
    aucs, rr = [], []
    for u in np.unique(qid):
        m = qid == u
        ys, ss = y[m], s[m]
        if ys.min() != ys.max():
            aucs.append(roc_auc_score(ys, ss))
        if ys.sum() > 0:
            order = np.argsort(-ss)
            rank = 1 + int(np.where(ys[order] == 1)[0][0])
            rr.append(1.0 / rank)
    return float(np.mean(aucs)), float(np.mean(rr))


def cluster_ci(y, s, srep, adv, qid, B=1000):
    qs = np.unique(qid)
    gains = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == t)[0] for t in take])
        m = adv[idx]
        ya, sa, ra = y[idx][m], s[idx][m], srep[idx][m]
        if ya.min() == ya.max():
            continue
        gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return float(lo), float(hi), len(gains)


def nested(dense, phi, y, s):
    idx = rng.permutation(len(y))
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]
    gains = []
    for a, b in [(0, 1), (1, 0)]:
        A, Bx = halves[a], halves[b]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(dense[A], y[A])
        art_b = m.predict_proba(dense[Bx])[:, 1]
        thr = np.quantile(art_b, 1 - y[Bx].mean())
        adv_b = (art_b > thr) != (y[Bx] == 1)
        srep_b = repair(phi[Bx], s[Bx])
        ya, sa, ra = y[Bx][adv_b], s[Bx][adv_b], srep_b[adv_b]
        gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
    return gains


def load_setting(name):
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
        scores = {"ce": np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"),
                  "qnli": np.load(f"{CACHE}/wikiqa_qnli_s_8000.npy"),
                  "stsb": np.load(f"{CACHE}/wikiqa_stsb_s_8000.npy")}
    elif name == "asnq":
        rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
        scores = {"ce": np.load(f"{CACHE}/asnq_ce_s_8000.npy")}
    elif name == "triviaqa":
        rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)]
        scores = {"ce": np.load(f"{CACHE}/triviaqa_ce_s.npy")}
    return rows, scores


out = {}
for name in ["wikiqa", "asnq", "triviaqa"]:
    rows, scores = load_setting(name)
    q, ans, y, dense, phi_full, phi_ans, phi_ov, qid = build(rows)
    art, adv = slice_from(dense, y)
    lin = lin_index(dense, y)
    res = {"n": len(y), "n_queries": int(len(np.unique(qid))),
           "n_adv": int(adv.sum())}
    for sc, s in scores.items():
        s = np.asarray(s, float)
        srep = repair(phi_full, s)
        ya, sa, ra = y[adv], s[adv], srep[adv]
        gain = roc_auc_score(ya, ra) - roc_auc_score(ya, sa)
        lo, hi, nb = cluster_ci(y, s, srep, adv, qid)
        pq_raw = per_query_metrics(y, s, qid)
        pq_rep = per_query_metrics(y, srep, qid)
        e1 = {"Rlin_prob_raw": abs(float(np.corrcoef(s, art)[0, 1])),
              "Rlin_prob_rep": abs(float(np.corrcoef(srep, art)[0, 1])),
              "Rlin_logit_raw": abs(float(np.corrcoef(s, lin)[0, 1])),
              "Rlin_logit_rep": abs(float(np.corrcoef(srep, lin)[0, 1]))}
        abl = {}
        for tag, ph in [("full", phi_full), ("answer_only", phi_ans),
                        ("overlap_only", phi_ov)]:
            rr = repair(ph, s)
            abl[tag] = float(roc_auc_score(y[adv], rr[adv])
                             - roc_auc_score(y[adv], s[adv]))
        res[sc] = {"slice_gain": float(gain),
                   "cluster_CI": [lo, hi], "cluster_B_kept": nb,
                   "perquery_AUC_raw": pq_raw[0], "perquery_AUC_rep": pq_rep[0],
                   "MRR_raw": pq_raw[1], "MRR_rep": pq_rep[1],
                   "certificate": e1, "ablation_gain": abl}
        if sc == "ce":
            res["nested_gains"] = nested(dense, phi_full, y, s)
    out[name] = res
    print(name, json.dumps(res, indent=1)[:600])

os.makedirs(os.path.join(HERE, "..", "notes"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "notes",
                                 "review_r2_experiments.json"), "w"), indent=1)
print("saved notes/review_r2_experiments.json")
