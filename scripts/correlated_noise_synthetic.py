"""Round-4 item 2 (priority 2): construct-measurement noise that VIOLATES the
independence definition, including the false-approval demonstration.

Setup: rho = 0.8 (construct-artifact collinear enough that repair truly HURTS:
Fig 1a shows 0.933 -> 0.737 there). True verdict: refuse.
Noise models on the construct measurement y:
  symmetric        : flip y uniformly at rate p            (independent)
  class_dependent  : flip only y=1 at rate 2p              (independent of phi)
  artifact_corr    : flip preferentially where the artifact channel DISAGREES
                     with the raw score's artifact loading (anti-artifact
                     noise) -> penalizes s, spares s_rep
  score_corr       : judge-style noise, flips where |s| is small (uncertain)
For each: measured full-set gain under y_noisy vs TRUE gain under y_true.
False approval = measured gain > 0 while true gain < 0.
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N = 4000
CV = KFold(5, shuffle=True, random_state=0)
RHO = 0.8


def auc(y, x):
    return float(roc_auc_score(y, x))


phi = RNG.normal(size=(N, 8))
w = RNG.normal(size=8) / np.sqrt(8)
a = phi @ w
a = (a - a.mean()) / a.std()
c = RHO * a + np.sqrt(1 - RHO**2) * RNG.normal(size=N)   # collinear construct
y_true = (c + 0.3 * RNG.normal(size=N) > 0).astype(int)
s = c + 1.2 * a + 0.3 * RNG.normal(size=N)
s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)

true_gain = auc(y_true, s_rep) - auc(y_true, s)
print(f"rho={RHO}  TRUE full-set gain = {true_gain:+.3f} (repair should be refused)")

results = {"rho": RHO, "true_gain": true_gain, "noise": {}}
P = 0.2

def flip(y, mask_prob):
    y2 = y.copy()
    fl = RNG.random(N) < mask_prob
    y2[fl] = 1 - y2[fl]
    return y2

# noise-probability profiles (mean flip rate ~= P for comparability)
prof = {
    "symmetric": np.full(N, P),
    "class_dependent": np.where(y_true == 1, 2 * P, 0.0),
    # anti-artifact noise: flips concentrated where the artifact strongly
    # agrees with y (a's sign matches the label) -> corrupts exactly the
    # examples on which the artifact-loaded raw score looks good
    "artifact_corr": np.where((a > 0) == (y_true == 1), 2 * P, 0.0),
    # judge noise: uncertain (mid-range) scores get noisy labels
    "score_corr": np.where(np.abs(s - s.mean()) < np.quantile(
        np.abs(s - s.mean()), 0.5), 2 * P, 0.0),
}
for name, pr in prof.items():
    y_n = flip(y_true, pr)
    g = auc(y_n, s_rep) - auc(y_n, s)
    results["noise"][name] = {
        "flip_rate": float(np.mean(y_n != y_true)),
        "measured_gain": g,
        "false_approval": bool(g > 0 > true_gain),
    }
    print(f"{name:16} flip={np.mean(y_n != y_true):.3f} "
          f"measured gain {g:+.3f}  false_approval={g > 0 > true_gain}")

dst = os.path.join(os.path.dirname(__file__),
                   "../notes/correlated_noise_2026-07-11.json")
with open(dst, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", dst)
