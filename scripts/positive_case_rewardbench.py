"""Positive-case testbed for proxy repair: RewardBench (reward margin = preference
construct + length/format artifact). Turn-key: plug real RewardBench per-example
scores into load_rewardbench(); a synthetic self-test runs now to validate the code.

Claim to demonstrate: residualizing the reward margin on length/format artifacts
RAISES preference agreement (construct) while closing the artifact channel — the
ρ≈0–0.3 "repairable" regime, complementing the RLC unrepairable anchor.

Run: python3 positive_case_rewardbench.py            # synthetic self-test
     python3 positive_case_rewardbench.py <data.json> # real RewardBench dump
"""
import sys, json, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


# ---------- repair operators ----------
def repair_linear(s, phi):
    """OLS residual (cross-fit): closes the LINEAR artifact channel (s_rep ⟂ phi cols)."""
    return s - cross_val_predict(LinearRegression(), phi, s, cv=5)

def repair_nonlinear(s, phi):
    """GBM residual (cross-fit): closes nonlinear artifact channel (no closed-form guarantee)."""
    return s - cross_val_predict(GradientBoostingRegressor(random_state=0), phi, s, cv=5)


def floor(x, target):
    """content-blind floor: can the artifact be recovered from the (scalar) score?"""
    return roc_auc_score(target, cross_val_predict(
        LinearRegression(), x.reshape(-1, 1), target.astype(float), cv=5))


def report(name, s, y, art_label, phi):
    sL = repair_linear(s, phi); sN = repair_nonlinear(s, phi)
    print(f"\n=== {name} ===")
    print(f"{'':16}{'pref-AUC(y)':>12}{'artifact-floor':>16}")
    for tag, x in [("raw margin", s), ("linear-repaired", sL), ("nonlinear-repaired", sN)]:
        print(f"{tag:16}{roc_auc_score(y, x):>12.3f}{floor(x, art_label):>16.3f}")
    print("expect: pref-AUC ↑ (or preserved) AND artifact-floor → chance = valid positive repair")


# ---------- real data adapter (fill this in with a RewardBench dump) ----------
def load_rewardbench(path):
    """Expect a json list of records, each with:
       'margin'      : reward(chosen) - reward(rejected)   [the score s]
       'correct'     : 1 if the RM ranked the truly-preferred response higher [preference construct y]
       'len_chosen','len_rejected' : token/char lengths    [artifact features phi]
       (optionally other format features: markdown count, bullet count, etc.)
    Source: RewardBench (allenai/reward-bench) per-example results, or run any RM over
    the pairs to produce margins. Lengths come from the response texts.
    """
    recs = json.load(open(path))
    s = np.array([r["margin"] for r in recs], float)
    y = np.array([int(r["correct"]) for r in recs])
    phi = np.array([[r["len_chosen"], r["len_rejected"],
                     r["len_chosen"] - r["len_rejected"]] for r in recs], float)
    art_label = (phi[:, 2] > 0).astype(int)          # "longer-chosen" as the artifact concept
    return s, y, art_label, phi


# ---------- synthetic self-test (runs now; proves the pipeline + the positive effect) ----------
def synthetic():
    rng = np.random.default_rng(0); N = 3000
    pref = rng.normal(size=N)                          # true preference latent
    length_gap = rng.normal(size=N)                    # length artifact latent (independent of pref)
    y = (pref > 0).astype(int)                         # construct label
    # reward margin = genuine preference + a length bias (RM prefers longer) + noise
    s = 1.0 * pref + 1.3 * length_gap + 0.3 * rng.normal(size=N)
    phi = np.c_[length_gap + 0.1 * rng.normal(size=N),
                length_gap ** 2 + 0.1 * rng.normal(size=N),
                rng.normal(size=N)]
    art_label = (length_gap > 0).astype(int)
    report("SYNTHETIC RewardBench-like (length artifact)", s, y, art_label, phi)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        s, y, art_label, phi = load_rewardbench(sys.argv[1])
        report(f"RewardBench: {sys.argv[1]}", s, y, art_label, phi)
    else:
        synthetic()
