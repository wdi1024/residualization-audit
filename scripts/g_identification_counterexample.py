"""Appendix L, 'What a positive G does not identify' (paper v176).

In the Gaussian model of that appendix the fitted component is (gamma*rho+beta)a,
not the artifact term beta*a. With beta=0 the score carries no artifact loading at
all, yet residualization still subtracts gamma*rho*a and raises alignment on the
idealized slice {ac<0}. Run: python3 g_identification_counterexample.py

Output (2026-09-08, seed 175, N=5e5): slice AUC 0.903 -> 0.987;
full-population AUC 0.973 -> 0.748; subtracted coefficient 0.799 (= gamma*rho)."""
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(175)
N, RHO, SIG, GAMMA, BETA = 500_000, 0.8, 0.3, 1.0, 0.0
a = rng.normal(size=N)
c = RHO * a + np.sqrt(1 - RHO**2) * rng.normal(size=N)
s = GAMMA * c + BETA * a + SIG * rng.normal(size=N)
coef = np.cov(s, a)[0, 1] / a.var()          # population projection of s on a
srep = s - coef * a
y = (c > 0).astype(int)
slice_ = (a * c) < 0                          # idealized slice of Property 4
print(f"subtracted coefficient {coef:.3f} (gamma*rho={GAMMA*RHO:.3f}, beta={BETA})")
print(f"slice AUC {roc_auc_score(y[slice_], s[slice_]):.3f} -> "
      f"{roc_auc_score(y[slice_], srep[slice_]):.3f}")
print(f"full  AUC {roc_auc_score(y, s):.3f} -> {roc_auc_score(y, srep):.3f}")
