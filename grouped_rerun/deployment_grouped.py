"""Question-grouped re-evaluation of the FROZEN deployment artifacts
(paper v172; reviewer round 4, item 'grouped verification').

Protocol, fixed in advance of seeing any grouped number:
  - The selected rule parameters are FROZEN at the recorded values
    (F2 |g|-switch t=90, F3 c-band q=10; F1 selected the identity).
    Nothing is re-selected.
  - The risk flag keeps its frozen FEATURE SPECIFICATION and hyperparameters;
    its logistic fit on pooled DEV is re-run once because its inputs
    (cross-fitted c_hat, g_hat, prob) change under grouped folds.
  - Cross-fitted objects use GroupKFold(5) with question groups, so no
    question spans a train/held-out boundary.
  - Point-check first: the row-level pipeline must reproduce the recorded
    numbers (rule deltas on SQuAD -0.0775/-0.0819; flag AUCs
    0.809/0.913/0.739) before any grouped number is reported.
Writes deployment_grouped.json next to this script.
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
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
    _, qid = np.unique(q, return_inverse=True)
    return y, dense, hstack([Xa, csr_matrix(dense)]).tocsr(), qid


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


def load_rows(name):
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        return [(r["question"], r["answer"], r["label"]) for r in d][:8000], \
            np.load(f"{CACHE}/wikiqa_ce_s_8000.npy")
    if name == "asnq":
        return json.load(open(f"{CACHE}/asnq_rows_8000.json")), \
            np.load(f"{CACHE}/asnq_ce_s_8000.npy")
    if name == "triviaqa":
        return [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)], \
            np.load(f"{CACHE}/triviaqa_ce_s.npy")
    if name == "squad":
        return squad_rows(), np.load(f"{CACHE}/squad_ce_s_8000.npy")
    if name == "duorc":
        return duorc_rows(), np.load(f"{CACHE}/duorc_ce_s_8000.npy")


def objects(name, grouped):
    rows, s = load_rows(name)
    y, dense, phi, qid = build(rows)
    s = np.asarray(s, float)
    if grouped:
        cv, kw = GroupKFold(5), {"groups": qid}
    else:
        cv, kw = KFold(5, shuffle=True, random_state=0), {}
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, **kw)
    art = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    prob = cross_val_predict(art, dense, y, cv=cv, method="predict_proba", **kw)[:, 1]
    chat = cross_val_predict(art, dense, y, cv=cv, method="decision_function", **kw)
    thr = np.quantile(prob, 1 - y.mean())
    adv = ((prob > thr) != (y == 1)).astype(int)
    return {"y": y, "s": s, "g": g, "c": chat, "prob": prob, "adv": adv, "qid": qid}


def apply_rule(o, fam, p):
    s, g, c = o["s"], o["g"], o["c"]
    srep = s - g
    if fam == "F2":
        t = np.percentile(np.abs(g - np.median(g)), p)
        m = np.abs(g - np.median(g)) > t
    elif fam == "F3":
        lo, hi = np.percentile(c, p), np.percentile(c, 100 - p)
        m = (c < lo) | (c > hi)
    out = s.copy()
    out[m] = (srep[m] - srep[m].mean()) / (srep[m].std() + 1e-9) \
        * s[~m].std() + s[~m].mean()
    return out


FROZEN = {"F2": 90, "F3": 10}


def cluster_ci(qid, fn, B=500):
    qs = np.unique(qid); vals = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == t)[0] for t in take])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rule_eval(o):
    out = {}
    for fam, p in FROZEN.items():
        u = apply_rule(o, fam, p)
        d = float(roc_auc_score(o["y"], u) - roc_auc_score(o["y"], o["s"]))

        def fn(idx, u=u):
            ya = o["y"][idx]
            if ya.min() == ya.max():
                return None
            return roc_auc_score(ya, u[idx]) - roc_auc_score(ya, o["s"][idx])
        out[fam] = {"param": p, "full_delta": round(d, 4),
                    "clusterCI": [round(x, 4) for x in cluster_ci(o["qid"], fn)]}
    return out


def flag_features(o):
    def zs(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([zs(o["c"]), np.abs(zs(o["c"])), zs(o["g"]),
                            np.abs(zs(o["g"])), zs(o["s"]), zs(o["s"] - o["g"]),
                            zs(o["prob"])])


def flag_eval(objs, DEV, HELD):
    Xdev = np.vstack([flag_features(objs[n]) for n in DEV])
    adv_dev = np.concatenate([objs[n]["adv"] for n in DEV])
    flag = LogisticRegression(max_iter=1000).fit(Xdev, adv_dev)
    out = {}
    for n in HELD:
        o = objs[n]
        pr = flag.predict_proba(flag_features(o))[:, 1]
        auc = float(roc_auc_score(o["adv"], pr))
        k = max(1, int(0.2 * len(pr)))
        top = np.argsort(-pr)[:k]
        prec20 = float(o["adv"][top].mean())
        base = float(o["adv"].mean())
        out[n] = {"flag_AUC": round(auc, 4), "base_rate": round(base, 4),
                  "precision_top20": round(prec20, 4),
                  "lift_top20": round(prec20 / base, 2)}
    return out


DEV, HELD = ["wikiqa", "asnq"], ["squad", "duorc", "triviaqa"]

print("== point-check (row-level reproduction) ==", flush=True)
row_objs = {n: objects(n, grouped=False) for n in DEV + HELD}
pc = rule_eval(row_objs["squad"])
print("squad row-level:", {f: v["full_delta"] for f, v in pc.items()}, flush=True)
assert abs(pc["F2"]["full_delta"] - (-0.0775)) < 0.003, pc["F2"]
assert abs(pc["F3"]["full_delta"] - (-0.0819)) < 0.003, pc["F3"]
fc = flag_eval(row_objs, DEV, HELD)
print("row-level flag AUCs:", {n: fc[n]["flag_AUC"] for n in HELD}, flush=True)
for n, want in [("squad", 0.8088), ("duorc", 0.9132), ("triviaqa", 0.7390)]:
    assert abs(fc[n]["flag_AUC"] - want) < 0.005, (n, fc[n])
print("point-check PASSED", flush=True)

print("== grouped evaluation (frozen rules and flag spec) ==", flush=True)
grp_objs = {n: objects(n, grouped=True) for n in DEV + HELD}
res = {"protocol": "GroupKFold(5) by question; rules frozen at F2=90, F3=10; "
                   "flag feature spec frozen, logistic refit once on pooled dev",
       "point_check_row_level": {"rules_squad": pc, "flag": fc},
       "rules_grouped": {n: rule_eval(grp_objs[n]) for n in HELD},
       "flag_grouped": flag_eval(grp_objs, DEV, HELD)}
json.dump(res, open(os.path.join(HERE, "deployment_grouped.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ["rules_grouped", "flag_grouped"]}, indent=1))
