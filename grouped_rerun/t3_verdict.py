"""Full committed-procedure verdict under the T.3 artifact-correlated-noise
construction (paper v172). Reproduces the correlated_noise_synthetic.py setup
exactly (same seed/order), asserts the recorded measured gains, then runs the
committed pipeline against the noisy label the auditor would hold:
  a-hat = logistic(phi -> y_noisy), slice from its errors, gates, R_lin,
  slice-gain bootstrap CI, final verdict."""
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N = 4000
CV = KFold(5, shuffle=True, random_state=0)
RHO = 0.8
phi = RNG.normal(size=(N, 8))
w = RNG.normal(size=8) / np.sqrt(8)
a = phi @ w; a = (a - a.mean()) / a.std()
c = RHO * a + np.sqrt(1 - RHO**2) * RNG.normal(size=N)
y_true = (c + 0.3 * RNG.normal(size=N) > 0).astype(int)
s = c + 1.2 * a + 0.3 * RNG.normal(size=N)
s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
true_gain = roc_auc_score(y_true, s_rep) - roc_auc_score(y_true, s)

def flip(y, pr):
    y2 = y.copy(); fl = RNG.random(N) < pr; y2[fl] = 1 - y2[fl]; return y2

P = 0.2
profiles = {
    "symmetric": np.full(N, P),
    "class_dependent": np.where(y_true == 1, 2 * P, 0.0),
    "artifact_corr": np.where((a > 0) == (y_true == 1), 2 * P, 0.0),
    "score_corr": np.where(np.abs(s - s.mean()) < np.quantile(np.abs(s - s.mean()), 0.5), 2 * P, 0.0),
}
print(f"true full-set gain {true_gain:+.4f} (recorded -0.191)")
assert abs(true_gain - (-0.191)) < 0.01, true_gain

for name, pr in profiles.items():
    y_n = flip(y_true, pr)
    g_full = roc_auc_score(y_n, s_rep) - roc_auc_score(y_n, s)
    # committed pipeline against the noisy label
    art = LogisticRegression(max_iter=1000)
    prob = cross_val_predict(art, phi, y_n, cv=CV, method="predict_proba")[:, 1]
    chat = cross_val_predict(art, phi, y_n, cv=CV, method="decision_function")
    Aa = roc_auc_score(y_n, prob)
    thr = np.quantile(prob, 1 - y_n.mean())
    adv = (prob > thr) != (y_n == 1)
    g_hat = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    R2 = 1 - ((s - g_hat) ** 2).sum() / ((s - s.mean()) ** 2).sum()
    Ds = roc_auc_score(y_n, s) - roc_auc_score(y_n[adv], s[adv])
    srep2 = s - g_hat
    Rlin = abs(np.corrcoef(srep2, chat)[0, 1])
    gain = roc_auc_score(y_n[adv], srep2[adv]) - roc_auc_score(y_n[adv], s[adv])
    boots = []
    rb = np.random.default_rng(1)
    idxs = np.where(adv)[0]
    for _ in range(500):
        ii = rb.choice(idxs, len(idxs), replace=True)
        if y_n[ii].min() == y_n[ii].max(): continue
        boots.append(roc_auc_score(y_n[ii], srep2[ii]) - roc_auc_score(y_n[ii], s[ii]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if Aa < 0.65: v = "screened out (router)"
    elif R2 < 0.05: v = "not validated (screen R2)"
    elif Ds < 0.03: v = "not validated (screen Dslice)"
    elif lo <= 0: v = "not validated (gain CI)"
    else: v = "VALIDATED" + (" [decorr withdrawn]" if Rlin > 0.05 else "")
    print(f"{name:15} gfull={g_full:+.3f} A(a)={Aa:.3f} R2={R2:.3f} Ds={Ds:+.3f} "
          f"Rlin={Rlin:.3f} gain={gain:+.3f} CI[{lo:+.3f},{hi:+.3f}]  -> {v}")
