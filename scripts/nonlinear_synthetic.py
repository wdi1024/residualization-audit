"""Nonlinear synthetic: why a LINEAR repair operator must refuse
nonlinear/structural artifacts (2026-07-10 feedback item 4).

Same generative family as the rho-sweep (rho = 0, i.e. repairable geometry
IF the channel were linear), but the artifact enters the SCORE through a
nonlinear function of the artifact features:
    linear    : a = w'phi                     (control; C1 passes)
    quadratic : a = (w'phi)^2 - E[..]
    xor       : a = sign(phi_1) * sign(phi_2)
    piecewise : a = 1[phi_1 > 0] * (v'phi)    (tree/branch-like)
score s = construct + 1.2 * a + noise; construct label independent of phi.

Predictions of the validity condition:
  - C1 (linear recoverability of the artifact target from phi via a LINEAR
    model) passes only for `linear`; fails for the others.
  - Linear repair helps only for `linear`; elsewhere it cannot remove the
    channel (floor stays high under a nonlinear probe) and/or removes
    construct-free variance only, so the adversarial-slice gain ~ 0.
  - A nonlinear residualizer (RF) recovers the nonlinear cases, at the cost
    of the linear certificate (reported for completeness).
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N, D = 4000, 8
CV = KFold(5, shuffle=True, random_state=0)


def auc(y, x, m=None):
    if m is not None:
        y, x = y[m], x[m]
    return float(roc_auc_score(y, x))


def make(kind):
    phi = RNG.normal(size=(N, D))
    w = RNG.normal(size=D) / np.sqrt(D)
    v = RNG.normal(size=D) / np.sqrt(D)
    if kind == "linear":
        a = phi @ w
    elif kind == "quadratic":
        a = (phi @ w) ** 2
    elif kind == "xor":
        a = np.sign(phi[:, 0]) * np.sign(phi[:, 1])
    elif kind == "piecewise":
        a = (phi[:, 0] > 0) * (phi @ v)
    a = (a - a.mean()) / (a.std() + 1e-12)
    c = RNG.normal(size=N)                      # construct, independent of phi
    y = (c + 0.3 * RNG.normal(size=N) > 0).astype(int)
    s = c + 1.2 * a + 0.3 * RNG.normal(size=N)
    return phi, a, y, s


def floor_probe(s_vec, z, probe):
    X = s_vec.reshape(-1, 1)
    p = cross_val_predict(probe, X, z, cv=CV, method="predict_proba")[:, 1]
    return auc(z, p)


results = {}
for kind in ("linear", "quadratic", "xor", "piecewise"):
    phi, a, y, s = make(kind)
    z = (a > np.median(a)).astype(int)          # artifact target (binarized)

    # C1: LINEAR recoverability of the artifact target from phi
    lin = cross_val_predict(LogisticRegression(max_iter=1000), phi, z,
                            cv=CV, method="predict_proba")[:, 1]
    c1_lin = auc(z, lin)

    # adversarial slice: artifact channel disagrees with construct label
    adv = (a > 0) != (y == 1)

    # linear repair
    s_lin = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    # nonlinear (RF) repair, for the discussion row
    s_rf = s - cross_val_predict(
        RandomForestRegressor(n_estimators=200, random_state=0), phi, s, cv=CV)

    row = {
        "C1_linear": c1_lin,
        "A_adv_raw": auc(y, s, adv),
        "A_adv_linear_repair": auc(y, s_lin, adv),
        "A_adv_rf_repair": auc(y, s_rf, adv),
        "floor_raw_rf_probe": floor_probe(s, z, RandomForestClassifier(
            n_estimators=200, random_state=0)),
        "floor_linrep_rf_probe": floor_probe(s_lin, z, RandomForestClassifier(
            n_estimators=200, random_state=0)),
        "floor_rfrep_rf_probe": floor_probe(s_rf, z, RandomForestClassifier(
            n_estimators=200, random_state=0)),
    }
    results[kind] = row
    print(kind, json.dumps(row, indent=1))

dst = os.path.join(os.path.dirname(__file__),
                   "../notes/nonlinear_synthetic_2026-07-10.json")
with open(dst, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", dst)
