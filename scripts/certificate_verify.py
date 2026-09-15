"""Round-4 items 1+3: certificate re-verification with a fixed definition of
the artifact object, and a nested (split-half) adversarial-slice protocol.

Certificate objects, per positive setting (SNLI / SICK / WikiQA):
  ghat  = cross-fitted ridge prediction of s from phi (what repair subtracts)
  ahat  = cross-fitted artifact-only prediction of the construct label (the
          continuous artifact CHANNEL; its thresholded version is the
          artifact DECISION used only for slicing)
  report |corr(s, ghat)|, |corr(s_rep, ghat)|   (orthogonality to removed comp.)
         |corr(s, ahat)|, |corr(s_rep, ahat)|   (linear recoverability of channel)
         decision-AUC floor raw/rep             (the stricter, non-guaranteed diag.)

Nested slice protocol (selection-bias check): shuffled split-half A/B.
  slice on B is defined by an artifact predictor trained ONLY on A;
  repair on B is cross-fitted ONLY within B; gain evaluated on B's slice.
  (and symmetrically A<->B). No fold ever both defines the slice and
  evaluates the repair.
"""
import json
import os
import subprocess

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix, issparse

CV = KFold(5, shuffle=True, random_state=0)
RNG = np.random.default_rng(0)
OUT = {}


def auc(y, x, m=None):
    if m is not None:
        y, x = y[m], x[m]
    return float(roc_auc_score(y, x))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------ data loaders
def load_snli():
    d = load_dataset("stanfordnlp/snli", split="validation")
    rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load("/tmp/snli_s_2400.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    art_model = LogisticRegression(max_iter=1000)
    return Xh, art_model, Xh, y, s, 0.5     # phi, art model, art features, thr


def load_sick():
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            "yangwang825%2Fsick&config=default&split=test")
    rows = []
    for off in range(0, 4906, 100):
        out = subprocess.run(["curl", "-sL", "--max-time", "60",
                              f"{base}&offset={off}&length=100"],
                             capture_output=True, check=True).stdout
        rows += [x["row"] for x in json.loads(out)["rows"]]
    pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
             for r in rows if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in pairs]
    y = np.array([c for _, _, c in pairs])
    s = np.load("/tmp/sick_s.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return Xh, LogisticRegression(max_iter=1000), Xh, y, s, 0.5


def load_wikiqa():
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load("/tmp/wikiqa_ce_s_8000.npy")

    def overlap(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / max(1, len(A | B))
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return phi, art_model, dense, y, s, "rate"   # thr = rate-calibrated


# ------------------------------------------------------------ certificate
def run_certificate(name, phi, art_model, art_X, y, s, thr_mode):
    ghat = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    s_rep = s - ghat
    ahat = cross_val_predict(art_model, art_X, y, cv=CV,
                             method="predict_proba")[:, 1]
    thr = 0.5 if thr_mode == 0.5 else np.quantile(ahat, 1 - y.mean())
    z = (ahat > thr).astype(int)

    def dec_floor(u):
        p = cross_val_predict(LogisticRegression(max_iter=1000),
                              u.reshape(-1, 1), z, cv=CV,
                              method="predict_proba")[:, 1]
        return auc(z, p)

    row = {
        "corr_s_ghat": corr(s, ghat), "corr_srep_ghat": corr(s_rep, ghat),
        "corr_s_ahat": corr(s, ahat), "corr_srep_ahat": corr(s_rep, ahat),
        "dec_floor_raw": dec_floor(s), "dec_floor_rep": dec_floor(s_rep),
    }
    OUT.setdefault(name, {})["certificate"] = row
    print(name, "certificate:", json.dumps(row, indent=1))
    return ahat, s_rep


# ------------------------------------------------------- nested slice check
def run_nested(name, phi, art_model, art_X, y, s, thr_mode):
    idx = RNG.permutation(len(y))
    halves = (idx[: len(y) // 2], idx[len(y) // 2:])
    gains = []
    for tr, ev in (halves, halves[::-1]):
        Xtr = art_X[tr] if issparse(art_X) else art_X[tr]
        m = art_model
        m.fit(Xtr, y[tr])
        pred_ev = m.predict_proba(art_X[ev])[:, 1]
        thr = 0.5 if thr_mode == 0.5 else np.quantile(pred_ev, 1 - y[ev].mean())
        adv = (pred_ev > thr) != (y[ev] == 1)
        phi_ev = phi[ev]
        s_ev = s[ev]
        s_rep_ev = s_ev - cross_val_predict(Ridge(alpha=1.0), phi_ev, s_ev, cv=CV)
        g = auc(y[ev], s_rep_ev, adv) - auc(y[ev], s_ev, adv)
        gains.append({"n_eval": int(len(ev)), "n_adv": int(adv.sum()),
                      "A_adv_raw": auc(y[ev], s_ev, adv),
                      "A_adv_rep": auc(y[ev], s_rep_ev, adv), "gain": g})
    OUT.setdefault(name, {})["nested"] = gains
    print(name, "nested:", json.dumps(gains, indent=1))


if __name__ == "__main__":
    for name, loader in (("SNLI", load_snli), ("SICK", load_sick),
                         ("WikiQA", load_wikiqa)):
        try:
            phi, art_model, art_X, y, s, thr = loader()
            run_certificate(name, phi, art_model, art_X, y, s, thr)
            run_nested(name, phi, art_model, art_X, y, s, thr)
        except Exception as e:
            OUT[name] = {"error": str(e)}
            print(name, "FAILED:", e)
    dst = os.path.join(os.path.dirname(__file__),
                       "../notes/certificate_nested_2026-07-11.json")
    with open(dst, "w") as f:
        json.dump(OUT, f, indent=1)
    print("wrote", dst)
