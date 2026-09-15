"""Does the screen move with the repair gain, or against it?

Re-runs the synthetic collinearity sweep of Figure 1 (identical generative model,
seed and N as scripts/proxy_repair_pilot.py) and additionally measures the two
screening statistics at every rho:

  C1            = AUC(y, a_hat)      label predictivity of phi   (gate ~0.65)
  R^2_{phi->s}  = cross-fitted       surface predictability of s (gate ~0.05)
  Delta_adv     = A(s) - A_adv(s)    degradation where a_hat errs (gate ~0.03)

against the repair gain the sweep already reports. If C1 rises with rho while the
full-population gain falls, C1 is not a repairability screen -- which is the claim
the revised Section 4 makes.
"""

import json

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

N = 4000


def run(rho, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)                                  # construct latent
    a = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=N)  # artifact latent
    phi = np.c_[a + 0.1 * rng.normal(size=(N,)),
                a ** 2 + 0.1 * rng.normal(size=(N,)),
                rng.normal(size=N)]                         # content-blind features
    s = 1.0 * z + 1.2 * a + 0.3 * rng.normal(size=N)        # evaluation score
    y = (z + 0.3 * rng.normal(size=N) > 0).astype(int)      # construct label

    ghat = cross_val_predict(LinearRegression(), phi, s, cv=5)
    s_rep = s - ghat

    # C1: label predictivity of phi (the artifact channel a_hat)
    ahat = cross_val_predict(LogisticRegression(max_iter=500), phi, y, cv=5,
                             method="predict_proba")[:, 1]
    c1 = roc_auc_score(y, ahat)

    # C2a: surface predictability of the score
    ss_res = float(((s - ghat) ** 2).sum())
    ss_tot = float(((s - s.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    # C2b: degradation on the slice where the artifact-only predictor errs
    zhat = (ahat > 0.5).astype(int)
    adv = zhat != y
    a_full = roc_auc_score(y, s)
    a_adv = roc_auc_score(y[adv], s[adv]) if 0 < y[adv].sum() < adv.sum() else float("nan")
    delta_adv = a_full - a_adv

    gain_full = roc_auc_score(y, s_rep) - a_full
    gain_adv = (roc_auc_score(y[adv], s_rep[adv]) - a_adv
                if 0 < y[adv].sum() < adv.sum() else float("nan"))
    return dict(rho=rho, c1=c1, r2=r2, delta_adv=delta_adv,
                align_raw=a_full, align_rep=roc_auc_score(y, s_rep),
                gain_full=gain_full, gain_adv=gain_adv, n_adv=int(adv.sum()))


rows = [run(r) for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

hdr = f"{'rho':>5} {'C1':>6} {'R2':>6} {'d_adv':>7} {'A(s)':>6} {'A(rep)':>7} {'gain_full':>10} {'gain_adv':>9} {'n_adv':>6}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['rho']:>5.1f} {r['c1']:>6.3f} {r['r2']:>6.3f} {r['delta_adv']:>7.3f} "
          f"{r['align_raw']:>6.3f} {r['align_rep']:>7.3f} {r['gain_full']:>+10.3f} "
          f"{r['gain_adv']:>+9.3f} {r['n_adv']:>6d}")

json.dump(rows, open("rho_sweep_screen.json", "w"), indent=1)
print("\n[check vs paper/rho_sweep.json] align_raw(0)=0.783, align_rep(0)=0.951 expected")
