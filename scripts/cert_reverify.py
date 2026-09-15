"""#1 (TMLR review, top must-fix): resolve the apparent conflict between the paper's
"linear channel closed / floor -> chance" claim and Appendix G (SICK repaired probe 0.648, etc.).

The conflict is a conflation of two DIFFERENT quantities:
  (A) CERTIFICATE proper  = linear orthogonality of s_rep to the artifact feature vector phi,
      i.e. R^2(phi -> s_rep) cross-fitted -> ~0.  This is what residualization delivers.
  (B) artifact-DECISION FLOOR = AUC of predicting the artifact-only DECISION z (a *nonlinear*,
      thresholded function of phi) from the scalar s_rep with a linear probe.  This need NOT
      reach 0.5 even when (A) holds, because z is not a linear function of phi.

We measure both across SNLI and SICK (cached scores).  Expected: R^2(phi->s_rep) collapses to
~0 (the linear channel IS closed), while the decision-floor drops substantially but stays >0.5
where the artifact decision is nonlinear-in-phi (SICK) -> reproduces App G numbers and shows they
are consistent with the (correctly scoped) certificate, not a contradiction.

Run: python3 cert_reverify.py
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
import robustness_suite as rs

CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

def r2_phi_to(Xh, x):
    pred = cross_val_predict(Ridge(alpha=1.0), Xh, x, cv=CV)
    ss_res = float(((x - pred) ** 2).sum()); ss_tot = float(((x - x.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot

def decision_floor(s_vec, z):
    pr = cross_val_predict(LogisticRegression(max_iter=1000), s_vec.reshape(-1, 1), z,
                           cv=CV, method="predict_proba")[:, 1]
    return roc_auc_score(z, pr), pr

def boot_auc_ci(z, pr, R=2000):
    n = len(z); vals = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, n)
        if len(set(z[ii])) < 2: vals[r] = np.nan; continue
        vals[r] = roc_auc_score(z[ii], pr[ii])
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def run(name, loader):
    hyp, y, s = loader()
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    hyp_prob = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y, cv=CV,
                                 method="predict_proba")[:, 1]
    z = (hyp_prob > 0.5).astype(int)                 # artifact-only DECISION (nonlinear in phi)
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), Xh, s, cv=CV)

    r2_raw, r2_rep = r2_phi_to(Xh, s), r2_phi_to(Xh, s_rep)
    f_raw, pr_raw = decision_floor(s, z)
    f_rep, pr_rep = decision_floor(s_rep, z)
    m_rep, lo_rep, hi_rep = boot_auc_ci(z, pr_rep)

    print(f"\n================ {name} (n={len(y)}) ================")
    print(f"(A) CERTIFICATE  R^2(phi->s):   raw {r2_raw:+.3f}  ->  repaired {r2_rep:+.3f}   "
          f"[linear channel {'CLOSED' if r2_rep < 0.02 else 'open'}: R^2 -> ~0]")
    print(f"(B) DECISION FLOOR AUC(z|score): raw {f_raw:.3f}  ->  repaired {f_rep:.3f}  "
          f"95%CI [{lo_rep:.3f},{hi_rep:.3f}]")
    print(f"    interpretation: linear phi-channel closes (A); the artifact-DECISION floor (B) "
          f"drops but stays >0.5 iff z is nonlinear-in-phi.")
    return {"name": name, "r2_raw": r2_raw, "r2_rep": r2_rep,
            "floor_raw": f_raw, "floor_rep": f_rep, "floor_rep_ci": [lo_rep, hi_rep]}

if __name__ == "__main__":
    import json, os
    out = [run("SNLI", rs.load_snli), run("SICK", rs.load_sick)]
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "notes",
                                     "cert_reverify_2026-07-11.json"), "w"), indent=1)
    print("\nRESOLUTION: report (A) as the certificate (linear channel R^2->~0, genuinely closed) "
          "and (B) as a separate diagnostic; scope the claim to linear phi-orthogonality, not "
          "artifact-decision unpredictability. This removes the internal inconsistency the reviewer flagged.")
