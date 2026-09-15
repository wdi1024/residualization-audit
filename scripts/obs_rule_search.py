"""Card 4: systematic search for a phi-only observable deployment rule.

Pre-declared rule families (decision-time inputs are phi-derived only):
  F1 shrinkage      s_lam = s - lam * g(phi),            lam in {0,.25,.5,.75,1}
  F2 |g|-switch     use s_rep iff |g - med(g)| > P_t,    t in {50,70,80,90}
  F3 c-band switch  use s_rep iff c_hat in top/bot q,    q in {10,20,30}%

Protocol: parameters selected on DEV = {WikiQA, ASNQ} by mean full-population
AUC delta over raw; the selected member of each family is then FROZEN and
evaluated once on HELD-OUT = {SQuAD, DuoRC} (full-population AUC delta with
query-clustered bootstrap CI). TriviaQA is reported as an additional
non-dev setting. Success = a frozen rule whose held-out full-population
delta is positive with CI excluding 0. All results reported either way.

Writes ../notes/obs_rule_search.json.
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
    rows, qid, seen = [], [], 0
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s_ in enumerate(sents) if ans in s_]
        if len(pos) != 1:
            continue
        for i, s_ in enumerate(sents):
            rows.append((r["question"], s_, 1 if i == pos[0] else 0)); qid.append(seen)
        seen += 1
        if len(rows) >= 8000:
            break
    return rows[:8000], np.array(qid[:8000])


def duorc_rows():
    from datasets import load_dataset
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows, qid, seen = [], [], 0
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
            rows.append((r["question"], s_, 1 if i == pos[0] else 0)); qid.append(seen)
        seen += 1
        if len(rows) >= 8000:
            break
    return rows[:8000], np.array(qid[:8000])


def setting(name):
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
        y, dense, phi = build(rows)
        return y, dense, phi, np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"), None
    if name == "asnq":
        rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
        y, dense, phi = build(rows)
        return y, dense, phi, np.load(f"{CACHE}/asnq_ce_s_8000.npy"), None
    if name == "triviaqa":
        rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)]
        y, dense, phi = build(rows)
        return y, dense, phi, np.load(f"{CACHE}/triviaqa_ce_s.npy"), None
    if name == "squad":
        rows, qid = squad_rows()
        y, dense, phi = build(rows)
        return y, dense, phi, np.load(f"{CACHE}/squad_ce_s_8000.npy"), qid
    if name == "duorc":
        rows, qid = duorc_rows()
        y, dense, phi = build(rows)
        return y, dense, phi, np.load(f"{CACHE}/duorc_ce_s_8000.npy"), qid


def objects(name):
    y, dense, phi, s, qid = setting(name)
    s = np.asarray(s, float)
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    art = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    chat = cross_val_predict(art, dense, y, cv=CV, method="decision_function")
    return {"y": y, "s": s, "g": g, "c": chat, "qid": qid}


def apply_rule(o, fam, p):
    s, g, c = o["s"], o["g"], o["c"]
    srep = s - g
    if fam == "F1":
        return s - p * g
    if fam == "F2":
        t = np.percentile(np.abs(g - np.median(g)), p)
        m = np.abs(g - np.median(g)) > t
        out = s.copy(); out[m] = (srep[m] - srep[m].mean()) / (srep[m].std() + 1e-9) \
            * s[~m].std() + s[~m].mean()
        return out
    if fam == "F3":
        lo, hi = np.percentile(c, p), np.percentile(c, 100 - p)
        m = (c < lo) | (c > hi)
        out = s.copy(); out[m] = (srep[m] - srep[m].mean()) / (srep[m].std() + 1e-9) \
            * s[~m].std() + s[~m].mean()
        return out


GRID = {"F1": [0.0, 0.25, 0.5, 0.75, 1.0], "F2": [50, 70, 80, 90],
        "F3": [10, 20, 30]}
DEV = ["wikiqa", "asnq"]
HELD = ["squad", "duorc"]

objs = {n: objects(n) for n in DEV + HELD + ["triviaqa"]}
res = {"dev_selection": {}, "frozen_heldout": {}}
frozen = {}
for fam, grid in GRID.items():
    scores = []
    for p in grid:
        deltas = [roc_auc_score(objs[n]["y"], apply_rule(objs[n], fam, p))
                  - roc_auc_score(objs[n]["y"], objs[n]["s"]) for n in DEV]
        scores.append((float(np.mean(deltas)), p))
    best = max(scores)
    frozen[fam] = best[1]
    res["dev_selection"][fam] = {"grid": [(p, round(d, 4)) for d, p in scores],
                                 "selected": best[1],
                                 "dev_mean_delta": round(best[0], 4)}

def cluster_ci_delta(o, u, B=500):
    qs = np.unique(o["qid"]); gains = []
    for _ in range(B):
        take = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(o["qid"] == t)[0] for t in take])
        ya = o["y"][idx]
        if ya.min() == ya.max():
            continue
        gains.append(roc_auc_score(ya, u[idx]) - roc_auc_score(ya, o["s"][idx]))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]

for n in HELD + ["triviaqa"]:
    o = objs[n]
    row = {}
    for fam, p in frozen.items():
        u = apply_rule(o, fam, p)
        d = float(roc_auc_score(o["y"], u) - roc_auc_score(o["y"], o["s"]))
        entry = {"param": p, "full_delta": round(d, 4)}
        if o["qid"] is not None:
            entry["clusterCI"] = [round(x, 4) for x in cluster_ci_delta(o, u)]
        row[fam] = entry
    res["frozen_heldout"][n] = row

json.dump(res, open(os.path.join(HERE, "..", "notes",
                                 "obs_rule_search.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
