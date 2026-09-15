"""Reviewer-robustness suite (2026-07-10 feedback, Must/Should items).

Covers, on the two real positives (SNLI, SICK) using cached scores:
  1. C1 threshold sensitivity (re-analysis across the scalar-C1 settings)
  2. C2 objectification: cross-fitted R^2(phi->s) and adversarial-slice
     degradation Delta_adv = A_full(s) - A_adv(s), per setting
  3. Alternative residualizers: OLS / Ridge / Lasso / ElasticNet
  4. Fold sensitivity: 3 / 5 / 10-fold cross-fitting
  5. Artifact-feature noise robustness: 10/20/30% Gaussian column noise
  6. Alternative artifact predictors: Logistic vs linear SVM (C1 + slice)
  7. Certificate robustness: linear / MLP / RF probes on s vs s_rep

Everything is CPU-only and uses the cached score arrays in /tmp.
Output: JSON to stdout + notes/robustness_suite_2026-07-10.json
"""
import json
import os
import subprocess

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (Ridge, LinearRegression, Lasso, ElasticNet,
                                  LogisticRegression)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
OUT = {}


def cv(k):
    return KFold(n_splits=k, shuffle=True, random_state=0)


def auc(y, x, mask=None):
    if mask is not None:
        y, x = y[mask], x[mask]
    return float(roc_auc_score(y, x))


# ---------------------------------------------------------------- loaders
def load_snli():
    d = load_dataset("stanfordnlp/snli", split="validation")
    rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load("/tmp/snli_s_2400.npy")
    return hyp, y, s


def load_sick():
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            "yangwang825%2Fsick&config=default&split=test")
    rows = []
    for off in range(0, 4906, 100):
        out = subprocess.run(["curl", "-sL", "--max-time", "60",
                              f"{base}&offset={off}&length=100"],
                             capture_output=True, check=True).stdout
        rows += [x["row"] for x in json.loads(out)["rows"]]
    # label semantics as recovered in sick_positive_run.py: entail=min-mean?
    # Reuse the deterministic outcome recorded there: ent label is the one
    # with the highest mean P(entail); recompute cheaply from cached s later,
    # but the binary subset construction only needs the two extreme labels.
    # sick_positive_run.py recovered entailment/contradiction from probe;
    # SICK labels: 0=ENTAILMENT,1=NEUTRAL,2=CONTRADICTION in this mirror.
    pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
             for r in rows if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in pairs]
    y = np.array([c for _, _, c in pairs])
    s = np.load("/tmp/sick_s.npy")
    assert len(s) == len(y), (len(s), len(y))
    # sanity: cached s must align with y (raw AUC ~0.967 per paper)
    print(f"[sick] raw AUC sanity: {auc(y, s):.3f} (expect ~0.967)")
    return hyp, y, s


# ------------------------------------------------------------- core pieces
def artifact_channel(Xh, y, k=5, kind="logistic"):
    if kind == "logistic":
        return cross_val_predict(LogisticRegression(max_iter=1000), Xh, y,
                                 cv=cv(k), method="predict_proba")[:, 1]
    if kind == "linsvm":
        dec = cross_val_predict(LinearSVC(C=1.0, max_iter=5000), Xh, y,
                                cv=cv(k), method="decision_function")
        return dec
    raise ValueError(kind)


def repair(Xh, s, reg, k=5):
    return s - cross_val_predict(reg, Xh, s, cv=cv(k))


def slice_gain(y, s, s_rep, adv):
    return auc(y, s_rep, adv) - auc(y, s, adv)


def certificate_probes(s_vec, z):
    """AUC of probes predicting artifact decision z from the 1-D score."""
    X = s_vec.reshape(-1, 1)
    res = {}
    probes = {
        "linear": LogisticRegression(max_iter=1000),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000,
                             random_state=0),
        "rf": RandomForestClassifier(n_estimators=200, random_state=0),
    }
    for name, p in probes.items():
        pr = cross_val_predict(p, X, z, cv=cv(5), method="predict_proba")[:, 1]
        res[name] = auc(z, pr)
    return res


def run_setting(name, hyp, y, s):
    print(f"\n================ {name} ================")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    R = {}

    # baseline artifact channel + adv slice (paper recipe)
    art = artifact_channel(Xh, y)
    adv = (art > 0.5) != (y == 1)
    c1 = auc(y, art)
    s_rep0 = repair(Xh, s, Ridge(alpha=1.0))
    R["baseline"] = {"C1": c1, "n_adv": int(adv.sum()),
                     "A_full_raw": auc(y, s), "A_adv_raw": auc(y, s, adv),
                     "A_full_rep": auc(y, s_rep0), "A_adv_rep": auc(y, s_rep0, adv)}
    print("baseline:", R["baseline"])

    # -- C2 objectification metrics
    s_hat = cross_val_predict(Ridge(alpha=1.0), Xh, s, cv=cv(5))
    ss_res = float(np.sum((s - s_hat) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2_phi_s = 1.0 - ss_res / ss_tot
    delta_adv = auc(y, s) - auc(y, s, adv)
    R["C2_metrics"] = {"R2_phi_to_s": r2_phi_s, "Delta_adv": delta_adv}
    print("C2 metrics:", R["C2_metrics"])

    # -- residualizer comparison
    regs = {"OLS": LinearRegression(),
            "Ridge(1.0)": Ridge(alpha=1.0),
            "Lasso(1e-3)": Lasso(alpha=1e-3, max_iter=20000),
            "ElasticNet(1e-3,0.5)": ElasticNet(alpha=1e-3, l1_ratio=0.5,
                                               max_iter=20000)}
    R["residualizers"] = {}
    for rn, reg in regs.items():
        sr = repair(Xh, s, reg)
        R["residualizers"][rn] = {"A_adv_rep": auc(y, sr, adv),
                                  "gain": slice_gain(y, s, sr, adv)}
        print(f"residualizer {rn}: {R['residualizers'][rn]}")

    # -- fold sensitivity
    R["folds"] = {}
    for k in (3, 5, 10):
        sr = repair(Xh, s, Ridge(alpha=1.0), k=k)
        artk = artifact_channel(Xh, y, k=k)
        advk = (artk > 0.5) != (y == 1)
        R["folds"][k] = {"A_adv_rep": auc(y, sr, advk),
                         "gain": slice_gain(y, s, sr, advk),
                         "n_adv": int(advk.sum())}
        print(f"folds {k}: {R['folds'][k]}")

    # -- noise robustness (Gaussian column noise on phi)
    Xd = np.asarray(Xh.todense())
    col_sd = Xd.std(axis=0) + 1e-12
    R["noise"] = {}
    for frac in (0.1, 0.2, 0.3):
        Xn = Xd + RNG.normal(0, frac, Xd.shape) * col_sd
        sr = repair(Xn, s, Ridge(alpha=1.0))
        R["noise"][frac] = {"A_adv_rep": auc(y, sr, adv),
                            "gain": slice_gain(y, s, sr, adv)}
        print(f"noise {frac}: {R['noise'][frac]}")

    # -- artifact predictor swap (logistic vs linear SVM)
    dec = artifact_channel(Xh, y, kind="linsvm")
    adv_svm = (dec > 0) != (y == 1)
    sr = repair(Xh, s, Ridge(alpha=1.0))
    R["artifact_predictor"] = {
        "logistic": {"C1": c1, "gain": slice_gain(y, s, sr, adv)},
        "linsvm": {"C1": auc(y, dec), "n_adv": int(adv_svm.sum()),
                   "gain": slice_gain(y, s, sr, adv_svm)}}
    print("artifact predictor:", R["artifact_predictor"])

    # -- certificate robustness (linear / MLP / RF probes, 1-D score)
    z = (art > 0.5).astype(int)
    R["certificate"] = {"raw": certificate_probes(s, z),
                        "repaired": certificate_probes(s_rep0, z)}
    print("certificate:", R["certificate"])
    return R


# --------------------------------------------------- C1 threshold re-analysis
def threshold_reanalysis():
    """Settings with a scalar C1 (artifact-only AUC), C2 verdict, and actual
    repair outcome, from the paper's Table 1 / notes."""
    settings = [
        # name, C1, C2(bool), actually_repaired(bool)
        ("SNLI", 0.708, True, True),
        ("SICK", 0.727, True, True),
        ("MNLI(off-dist)", 0.617, False, False),
        ("Toxicity", 0.557, False, False),
        ("QQP", 0.636, False, False),
        ("ANLI", 0.556, False, False),
        ("HellaSwag", 0.590, False, False),
        ("SWAG", 0.563, False, False),
    ]
    rows = []
    for t in (0.55, 0.60, 0.65, 0.70, 0.75):
        for mode in ("C1_only", "C1_and_C2"):
            tp = fp = fn = tn = 0
            for _, c1, c2, actual in settings:
                pred = (c1 >= t) and (c2 if mode == "C1_and_C2" else True)
                tp += pred and actual
                fp += pred and not actual
                fn += (not pred) and actual
                tn += (not pred) and (not actual)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            agree = (tp + tn) / len(settings)
            rows.append({"threshold": t, "mode": mode, "predicted_pos": tp + fp,
                         "precision": prec, "recall": rec, "agreement": agree})
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    OUT["threshold_reanalysis"] = threshold_reanalysis()

    hyp, y, s = load_snli()
    OUT["SNLI"] = run_setting("SNLI", hyp, y, s)

    try:
        hyp, y, s = load_sick()
        OUT["SICK"] = run_setting("SICK", hyp, y, s)
    except Exception as e:  # network-dependent
        OUT["SICK"] = {"error": str(e)}
        print("SICK failed:", e)

    dst = os.path.join(os.path.dirname(__file__),
                       "../notes/robustness_suite_2026-07-10.json")
    with open(dst, "w") as f:
        json.dump(OUT, f, indent=1)
    print("\nwrote", dst)
