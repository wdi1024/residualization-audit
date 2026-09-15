"""Card D: an observable artifact-risk FLAG (deployment triage, not repair).

The negative results show no phi-only rule improves the score on observable
populations. This reframes the deliverable: instead of changing the score,
emit a calibrated per-example WARNING that the example likely lies in the
artifact-suspect slice (where the artifact-only predictor errs). Flag inputs
are deployment-observable: the channel index c_hat, the removed component
g_hat, the raw score s, and simple transforms. The flag model is fit on the
DEV settings (WikiQA+ASNQ, per-setting standardized features, y used only in
dev fitting), FROZEN, and evaluated once on the held-out settings
(SQuAD, DuoRC): AUC against true slice membership, precision at top-K%,
and lift over base rate.

Writes ../notes/obs_risk_flag.json.
"""
import json
import os
import re

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


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def build(rows):
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    return y, dense, hstack([Xa, csr_matrix(dense)]).tocsr()


def squad_rows():
    from datasets import load_dataset
    d = load_dataset("squad", split="validation")
    rows = []
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s_ in enumerate(sents) if ans in s_]
        if len(pos) != 1:
            continue
        for i, s_ in enumerate(sents):
            rows.append((r["question"], s_, 1 if i == pos[0] else 0))
        if len(rows) >= 8000:
            break
    return rows[:8000]


def duorc_rows():
    from datasets import load_dataset
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows = []
    for r in d:
        if r["no_answer"] or not r["answers"]:
            continue
        sents = sent_split(r["plot"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"][0]
        if len(ans.split()) > 12 or not ans.strip():
            continue
        pos = [i for i, s_ in enumerate(sents) if ans in s_]
        if len(pos) != 1:
            continue
        for i, s_ in enumerate(sents):
            rows.append((r["question"], s_, 1 if i == pos[0] else 0))
        if len(rows) >= 8000:
            break
    return rows[:8000]


def objects(name):
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
        s = np.load(f"{CACHE}/wikiqa_ce_s_8000.npy")
    elif name == "asnq":
        rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
        s = np.load(f"{CACHE}/asnq_ce_s_8000.npy")
    elif name == "squad":
        rows = squad_rows(); s = np.load(f"{CACHE}/squad_ce_s_8000.npy")
    elif name == "duorc":
        rows = duorc_rows(); s = np.load(f"{CACHE}/duorc_ce_s_8000.npy")
    elif name == "triviaqa":
        rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)]
        s = np.load(f"{CACHE}/triviaqa_ce_s.npy")
    y, dense, phi = build(rows)
    s = np.asarray(s, float)
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    art = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    prob = cross_val_predict(art, dense, y, cv=CV, method="predict_proba")[:, 1]
    chat = cross_val_predict(art, dense, y, cv=CV, method="decision_function")
    thr = np.quantile(prob, 1 - y.mean())
    adv = ((prob > thr) != (y == 1)).astype(int)
    # deployment-observable flag features, standardized within setting
    def zs(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    X = np.column_stack([zs(chat), np.abs(zs(chat)), zs(g), np.abs(zs(g)),
                         zs(s), zs(s - g), zs(prob)])
    # cluster ids for query-clustered bootstrap: group by question text
    _, qid = np.unique([r[0] for r in rows], return_inverse=True)
    return X, adv, y, qid


DEV = ["wikiqa", "asnq"]
HELD = ["squad", "duorc", "triviaqa"]
objs = {n: objects(n) for n in DEV + HELD}

Xdev = np.vstack([objs[n][0] for n in DEV])
adv_dev = np.concatenate([objs[n][1] for n in DEV])
flag = LogisticRegression(max_iter=1000).fit(Xdev, adv_dev)   # frozen

res = {"dev_fit": {"n": int(len(adv_dev)),
                   "base_rate": float(adv_dev.mean()),
                   "in_sample_AUC": float(roc_auc_score(
                       adv_dev, flag.predict_proba(Xdev)[:, 1]))}}
def cluster_boot(qid, fn, B=1000):
    qs = np.unique(qid)
    vals = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == t)[0] for t in take])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


for n in HELD + DEV:
    X, adv, y, qid = objs[n]
    p = flag.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(adv, p))

    def auc_fn(idx):
        a = adv[idx]
        return roc_auc_score(a, p[idx]) if a.min() != a.max() else None

    row = {"base_rate": float(adv.mean()), "flag_AUC": auc,
           "flag_AUC_clusterCI": cluster_boot(qid, auc_fn)}
    for k in (10, 20):
        t = np.percentile(p, 100 - k)
        m = p >= t
        row[f"precision_top{k}"] = float(adv[m].mean())
        row[f"lift_top{k}"] = float(adv[m].mean() / max(1e-9, adv.mean()))

        def prec_fn(idx, thr_=t):
            mm = p[idx] >= thr_
            return adv[idx][mm].mean() if mm.sum() else None

        row[f"precision_top{k}_clusterCI"] = cluster_boot(qid, prec_fn)
    res[n] = row
json.dump(res, open(os.path.join(HERE, "..", "notes",
                                 "obs_risk_flag.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
