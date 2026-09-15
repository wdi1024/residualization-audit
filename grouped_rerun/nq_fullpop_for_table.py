"""Full-population change for the NQ row of the held-out summary table
(paper v160, tab:heldout). Reuses the frozen recipe of nq_dual_reading.py
from the released caches; asserts the recorded slice gain reproduces
before reporting anything new.

Output (2026-09-06 run): point-check gain_adv 0.2561 == artifact 0.2561;
A_full 0.7829 -> 0.7171 (delta -0.0658, query-clustered 95% CI
[-0.0788, -0.0538], 1000 resamples, seed 0)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "../cache/nq_rows.json")))
s = np.load(os.path.join(HERE, "../cache/nq_ce_s.npy"))
q = [r[0] for r in rows]; ans = [r[1] for r in rows]
y = np.array([r[2] for r in rows]); qid = np.array([r[3] for r in rows])
sys.path.insert(0, HERE)
from review_r4_experiments import qa_features
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
phi, dense = qa_features(q, ans)
cv = GroupKFold(5)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
art = cross_val_predict(pipe, dense, y, cv=cv, groups=qid, method="predict_proba")[:, 1]
thr = np.quantile(art, 1 - y.mean())
adv = (art > thr) != (y == 1)
srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, groups=qid)
gain_adv = roc_auc_score(y[adv], srep[adv]) - roc_auc_score(y[adv], s[adv])
assert abs(gain_adv - 0.2561) < 5e-4, f"point-check failed: {gain_adv:.4f}"
print(f"point-check ok: gain_adv={gain_adv:.4f}")
print(f"A_full raw/rep = {roc_auc_score(y, s):.4f} / {roc_auc_score(y, srep):.4f}")
rng = np.random.default_rng(0); uq = np.unique(qid)
idx = {u: np.where(qid == u)[0] for u in uq}
ds = []
for _ in range(1000):
    ii = np.concatenate([idx[u] for u in rng.choice(uq, len(uq), replace=True)])
    if len(np.unique(y[ii])) < 2: continue
    ds.append(roc_auc_score(y[ii], srep[ii]) - roc_auc_score(y[ii], s[ii]))
print("full-pop delta CI:", np.percentile(ds, [2.5, 97.5]).round(4))
