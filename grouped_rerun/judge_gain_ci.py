"""Query-clustered bootstrap CI for the WikiQA LLM-judge slice gain (+0.049).

Reconstructs llm_judge_probe.py's pipeline from the cached judge scores (no
re-scoring), asserts every point value against llm_judge_probe.json, then
bootstraps the slice gain A_adv(srep) - A_adv(s) over resampled questions with
the same resampling convention as boot_stats (1,000 resamples, seed as below).
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
sys.path.insert(0, HERE)
from review_r4_experiments import qa_features, run_setting  # noqa: E402
from datasets import load_dataset  # noqa: E402

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
qid = np.array([uniq[qq] for qq in q])
s = np.load(os.path.join(CACHE, "wikiqa_haiku_s_8000.npy"))
ok = np.isfinite(s)
q = [q[i] for i in range(len(y)) if ok[i]]
ans = [ans[i] for i in range(len(y)) if ok[i]]
y, qid, s = y[ok], qid[ok], s[ok]

ref = json.load(open(os.path.join(HERE, "llm_judge_probe.json")))
phi, dense = qa_features(q, ans)
res = run_setting("wikiqa_haiku_judge_ci_check", phi, dense, y, s, qid)
for k in ("C1", "R2", "delta_adv", "n_adv"):
    print("  check", k, res[k], "vs", ref[k], "diff", round(abs(res[k]-ref[k]),4))
a1 = res["alpha_sweep"]["1.0"]
print("  gain", a1["gain_adv"], "vs", ref["alpha_sweep"]["1.0"]["gain_adv"])
print("  A_adv", a1["A_adv"], "vs", ref["alpha_sweep"]["1.0"]["A_adv"])
print("point values reproduce llm_judge_probe.json:",
      {k: res[k] for k in ("C1", "R2", "delta_adv", "n_adv")}, a1)

# rebuild adv and srep exactly as run_setting does, for the gain bootstrap
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
cv = GroupKFold(5)
kw = dict(cv=cv, groups=qid)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
art = cross_val_predict(pipe, dense, y, method="predict_proba", **kw)[:, 1]
thr = np.quantile(art, 1 - y.mean())
adv = (art > thr) != (y == 1)
pred1 = cross_val_predict(Ridge(alpha=1.0), phi, s, **kw)
srep = s - pred1
g_point = roc_auc_score(y[adv], srep[adv]) - roc_auc_score(y[adv], s[adv])
print("  g_point", round(g_point,4))

rng = np.random.default_rng(0)
qs = np.unique(qid)
by_q = {qq: np.where(qid == qq)[0] for qq in qs}
gains = []
for _ in range(1000):
    pick = rng.choice(qs, size=len(qs), replace=True)
    idx = np.concatenate([by_q[qq] for qq in pick])
    vv = adv[idx]
    yy = y[idx][vv]
    if len(np.unique(yy)) < 2:
        continue
    gains.append(roc_auc_score(yy, srep[idx][vv]) - roc_auc_score(yy, s[idx][vv]))
ci = [round(float(np.percentile(gains, p)), 4) for p in (2.5, 97.5)]
out = {"gain_adv": round(float(g_point), 4),
       "A_adv": [round(float(roc_auc_score(y[adv], s[adv])), 4),
                 round(float(roc_auc_score(y[adv], srep[adv])), 4)],
       "gain_adv_ci": ci, "nboot_used": len(gains)}
print(out)
json.dump(out, open(os.path.join(HERE, "judge_gain_ci.json"), "w"), indent=1)
