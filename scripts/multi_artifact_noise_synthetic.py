"""Round-3 feedback items 4+5: multiple-artifact synthetic + construct-label
noise robustness (2026-07-10 evening).

A. Multiple artifacts. s = c + 1.0*aA + 0.8*aB + eps, rho(c,aA)=rho(c,aB)=0.
   Three variants:
     both_linear_observed : aA, aB linear, phi = [phiA, phiB]
     B_unobserved         : both linear, phi covers A only -> partial repair,
                            B's floor stays open (the certificate catches it)
     B_nonlinear          : aA linear, aB = XOR (not linearly recoverable)
   Reported per variant: adversarial-slice alignment raw -> repaired, linear
   floor for each artifact's decision after repair.

B. Construct-label noise. The SNLI-like linear synthetic (rho=0), with the
   construct LABEL y flipped at 0/10/20/30% before validation. Question: does
   the adversarial-slice gain (and hence the validation verdict) survive a
   noisy construct measurement?
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N = 4000
CV = KFold(5, shuffle=True, random_state=0)


def auc(y, x, m=None):
    if m is not None:
        y, x = y[m], x[m]
    return float(roc_auc_score(y, x))


def lin_floor(s_vec, z):
    p = cross_val_predict(LogisticRegression(max_iter=1000),
                          s_vec.reshape(-1, 1), z, cv=CV,
                          method="predict_proba")[:, 1]
    return auc(z, p)


# ---------------------------------------------------------------- part A
def multi_artifact(variant):
    phiA = RNG.normal(size=(N, 6))
    phiB = RNG.normal(size=(N, 6))
    wA = RNG.normal(size=6) / np.sqrt(6)
    wB = RNG.normal(size=6) / np.sqrt(6)
    aA = phiA @ wA
    if variant == "B_nonlinear":
        aB = np.sign(phiB[:, 0]) * np.sign(phiB[:, 1])
    else:
        aB = phiB @ wB
    aA = (aA - aA.mean()) / aA.std()
    aB = (aB - aB.mean()) / aB.std()
    c = RNG.normal(size=N)
    y = (c + 0.3 * RNG.normal(size=N) > 0).astype(int)
    s = c + 1.0 * aA + 0.8 * aB + 0.3 * RNG.normal(size=N)

    phi = phiA if variant == "B_unobserved" else np.hstack([phiA, phiB])
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)

    adv = ((aA + 0.8 * aB) > 0) != (y == 1)   # combined channel misleads
    zA = (aA > 0).astype(int)
    zB = (aB > 0).astype(int)
    return {
        "A_adv_raw": auc(y, s, adv), "A_adv_rep": auc(y, s_rep, adv),
        "floorA_rep": lin_floor(s_rep, zA), "floorB_rep": lin_floor(s_rep, zB),
        "floorA_raw": lin_floor(s, zA), "floorB_raw": lin_floor(s, zB),
    }


# ---------------------------------------------------------------- part B
def label_noise():
    phi = RNG.normal(size=(N, 8))
    w = RNG.normal(size=8) / np.sqrt(8)
    a = phi @ w
    a = (a - a.mean()) / a.std()
    c = RNG.normal(size=N)
    y_clean = (c + 0.3 * RNG.normal(size=N) > 0).astype(int)
    s = c + 1.2 * a + 0.3 * RNG.normal(size=N)
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    adv = (a > 0) != (y_clean == 1)   # slice fixed by the true geometry
    out = {}
    for frac in (0.0, 0.1, 0.2, 0.3):
        y = y_clean.copy()
        flip = RNG.random(N) < frac
        y[flip] = 1 - y[flip]
        advn = (a > 0) != (y == 1)    # slice as an auditor would compute it
        out[frac] = {
            "A_adv_raw": auc(y, s, advn), "A_adv_rep": auc(y, s_rep, advn),
            "gain": auc(y, s_rep, advn) - auc(y, s, advn),
            "A_full_raw": auc(y, s), "A_full_rep": auc(y, s_rep),
        }
    return out


results = {"multi_artifact": {}, "label_noise": label_noise()}
for v in ("both_linear_observed", "B_unobserved", "B_nonlinear"):
    results["multi_artifact"][v] = multi_artifact(v)
    print(v, json.dumps(results["multi_artifact"][v], indent=1))
print("label_noise", json.dumps(results["label_noise"], indent=1))

dst = os.path.join(os.path.dirname(__file__),
                   "../notes/multi_artifact_noise_2026-07-10.json")
with open(dst, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", dst)
