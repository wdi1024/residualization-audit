"""Internal review round 5, item C4: can the observable risk flag be used as
a SWITCH (residualized score on the top flagged quantile, raw score
elsewhere) rather than a warning?

The 12-member phi-only rule family (obs_rule_search.py) did not contain this
rule because the flag consumes score-derived features beyond phi; the flag
itself localizes the artifact-suspect slice at AUC 0.74-0.91 (obs_risk_flag.py),
so the natural question is whether switching to s_rep exactly there beats
both endpoints. Protocol: same flag as obs_risk_flag.py (logistic on
observable features, fit on WikiQA+ASNQ dev, frozen), same row construction
and row-level cross-fitting as that appendix suite; the switch
u_q(x) = s_rep(x) if flag(x) in top q% else s(x) is evaluated on the
held-out settings (SQuAD, DuoRC, TriviaQA) for q in {10, 20, 30, 50}, against
the q=0 (raw) and q=100 (full residualization) endpoints, on the full
population and on the true label-defined slice.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

# reuse obs_risk_flag's loaders/feature builders without executing its main
import importlib.util
spec = importlib.util.spec_from_file_location(
    "orf", os.path.join(HERE, "..", "scripts", "obs_risk_flag.py"))
orf = importlib.util.module_from_spec(spec)
src = open(os.path.join(HERE, "..", "scripts", "obs_risk_flag.py")).read()
# keep everything up to the module-level experiment (starts at "DEV = ")
head = src.split("\nDEV = ")[0]
exec(compile(head, "obs_risk_flag_head", "exec"), orf.__dict__)

np = orf.np
from sklearn.linear_model import LogisticRegression


def objects_with_scores(name):
    """orf.objects plus the raw and residualized scores themselves."""
    X, adv, y, qid = orf.objects(name)
    # rebuild s and g exactly as orf.objects does internally
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
        s = np.load(f"{orf.CACHE}/wikiqa_ce_s_8000.npy")
    elif name == "asnq":
        rows = json.load(open(f"{orf.CACHE}/asnq_rows_8000.json"))
        s = np.load(f"{orf.CACHE}/asnq_ce_s_8000.npy")
    elif name == "squad":
        rows = orf.squad_rows()
        s = np.load(f"{orf.CACHE}/squad_ce_s_8000.npy")
    elif name == "duorc":
        rows = orf.duorc_rows()
        s = np.load(f"{orf.CACHE}/duorc_ce_s_8000.npy")
    elif name == "triviaqa":
        rows = [list(r) for r in np.load(f"{orf.CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)]
        s = np.load(f"{orf.CACHE}/triviaqa_ce_s.npy")
    yb, dense, phi = orf.build(rows)
    s = np.asarray(s, float)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=orf.CV)
    return X, adv, y, qid, s, s - g


DEV = ["wikiqa", "asnq"]
HELD = ["squad", "duorc", "triviaqa"]
objs = {n: objects_with_scores(n) for n in DEV + HELD}

Xdev = np.vstack([objs[n][0] for n in DEV])
adv_dev = np.concatenate([objs[n][1] for n in DEV])
flag = LogisticRegression(max_iter=1000).fit(Xdev, adv_dev)  # frozen

res = {}
for n in HELD:
    X, adv, y, qid, s, srep = objs[n]
    p = flag.predict_proba(X)[:, 1]
    advb = adv.astype(bool)
    A = lambda m, u: round(float(roc_auc_score(y[m], u[m])), 4)
    full = np.ones(len(y), bool)
    row = {"raw": {"full": A(full, s), "slice": A(advb, s)},
           "full_repair": {"full": A(full, srep), "slice": A(advb, srep)}}
    for q in (10, 20, 30, 50):
        cut = np.quantile(p, 1 - q / 100)
        sw = np.where(p >= cut, srep, s)
        row[f"switch_top{q}"] = {"full": A(full, sw), "slice": A(advb, sw),
                                 "n_switched": int((p >= cut).sum())}
    res[n] = row
    print(n, json.dumps(row, indent=1), flush=True)

json.dump(res, open(os.path.join(HERE, "risk_flag_switch.json"), "w"), indent=1)
print("wrote risk_flag_switch.json")
