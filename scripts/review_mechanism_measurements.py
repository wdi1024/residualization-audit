"""Two mechanism measurements added in response to reviewer questions.

(1) Why linear residualization can RAISE nonlinear recoverability on SNLI
    (Appendix: "Why subtraction can raise nonlinear recoverability", Table
    tab:relocation). Reproduces the 0.581 -> 0.827 effect and shows the cause:
    the monotone/mean component is removed, while a non-monotone,
    interval-coded remainder survives because the audited score is saturated.

(2) Why the ASNQ (QNLI) replication passes both screens with a null gain
    (Appendix: falsification conditions). Measures the slice-conditional
    correlation of the removed component with the construct label, which the
    gates do not read.

Both are CPU-only and use the released caches. Requires the SNLI/ASNQ caches
to be visible at /tmp (the loaders in robustness_suite.py expect them there).
"""

import json
import os

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
BANDS = [(-1.10, -0.50), (-0.50, -0.15), (-0.15, 0.15), (0.15, 0.50), (0.50, 1.10)]


def snli_relocation():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import robustness_suite as RS

    hyp, y, s = RS.load_snli()
    y, s = np.asarray(y), np.asarray(s)
    phi = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    kf = KFold(5, shuffle=True, random_state=0)

    ahat, g = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in kf.split(np.arange(len(y))):
        ahat[te] = LogisticRegression(max_iter=2000).fit(phi[tr], y[tr]).predict_proba(phi[te])[:, 1]
        g[te] = RidgeCV(alphas=[0.1, 1, 10]).fit(phi[tr], s[tr]).predict(phi[te])
    srep, z = s - g, (ahat > 0.5).astype(int)

    def rf_auc(u):
        u = u.reshape(-1, 1)
        p = np.zeros(len(u))
        for tr, te in kf.split(u):
            rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2)
            p[te] = rf.fit(u[tr], z[tr]).predict_proba(u[te])[:, 1]
        return roc_auc_score(z, p)

    out = {
        "rf_auc_raw": round(rf_auc(s), 3),
        "rf_auc_residual": round(rf_auc(srep), 3),
        "linear_auc_raw": round(roc_auc_score(z, s), 3),
        "linear_auc_residual": round(roc_auc_score(z, srep), 3),
        "mean_gap_raw": round(float(s[z == 1].mean() - s[z == 0].mean()), 3),
        "mean_gap_residual": round(float(srep[z == 1].mean() - srep[z == 0].mean()), 3),
        "sd_ratio_raw": round(float(s[z == 1].std() / s[z == 0].std()), 2),
        "sd_ratio_residual": round(float(srep[z == 1].std() / srep[z == 0].std()), 2),
        "saturation_below_0.05": round(float((s < 0.05).mean()), 3),
        "saturation_above_0.95": round(float((s > 0.95).mean()), 3),
        "bands": [],
    }
    for lo, hi in BANDS:
        m = (srep >= lo) & (srep < hi)
        out["bands"].append({"band": [lo, hi], "n": int(m.sum()),
                             "p_z1": round(float(z[m].mean()), 3)})
    return out


def asnq_null_gain():
    rows = json.load(open(os.path.join(CACHE, "asnq_rows_8000.json")))
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])

    def overlap(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / max(1, len(A | B))

    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    phi = hstack([TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans),
                  csr_matrix(dense)]).tocsr()

    # stable group ids: hash() is salted per process, which would make the
    # fold assignment (and therefore these numbers) irreproducible.
    uniq = {t: i for i, t in enumerate(dict.fromkeys(q))}
    groups = np.array([uniq[t] for t in q])
    cv = GroupKFold(5)
    art = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                            dense, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    adv = (art > np.quantile(art, 1 - y.mean())) != (y == 1)

    out = {"n_adv": int(adv.sum()), "scorers": {}}
    for name, f in (("qnli", "asnq_qnli_s_8000.npy"), ("minilm", "asnq_ce_s_8000.npy")):
        s = np.load(os.path.join(CACHE, f))
        g = cross_val_predict(RidgeCV(alphas=[0.1, 1, 10]), phi, s, cv=cv, groups=groups)
        out["scorers"][name] = {
            "corr_removed_y_full": round(float(np.corrcoef(g, y)[0, 1]), 3),
            "corr_removed_y_slice": round(float(np.corrcoef(g[adv], y[adv])[0, 1]), 3),
        }
    return out


if __name__ == "__main__":
    res = {"snli_relocation": snli_relocation(), "asnq_null_gain": asnq_null_gain()}
    print(json.dumps(res, indent=2))
    with open(os.path.join(CACHE, "..", "review_mechanism_measurements.json"), "w") as fh:
        json.dump(res, fh, indent=2)
