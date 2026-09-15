"""Card 3: a nonlinear-closure eraser and its measured certificate.

The linear residualizer closes only the linear index; nonlinear probes still
recover the artifact decision (paper, Appendix G). The natural stronger
primitive is the CONDITIONAL-QUANTILE eraser: replace s by its within-bin
quantile, binning on the channel index c_hat. In the population this makes
s_q independent of (binned) c_hat, so EVERY function of c_hat---including the
thresholded decision z and any nonlinear probe of it---loses predictability
from s_q alone. The open question is the price: how much construct alignment
survives, relative to linear residualization?

For each positive setting we report:
  - nonlinear recoverability of z from the score alone (RF probe AUC):
    raw / linear-residual / quantile-erased   (the new certificate object)
  - linear index correlation (the old certificate) for all three
  - construct alignment A(y, .) on the full set and the adversarial slice
    for all three  (the measured certificate--validity trade-off)

Writes ../notes/quantile_eraser.json.
"""
import json
import os
import subprocess

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)
QBINS = 20


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def qa_build(rows):
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    return y, dense, hstack([Xa, csr_matrix(dense)]).tocsr()


def load(name):
    if name == "wikiqa":
        from datasets import load_dataset
        d = load_dataset("microsoft/wiki_qa", split="train")
        rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
        y, dense, phi = qa_build(rows)
        return y, dense, phi, np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"), "rate"
    if name == "asnq":
        rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
        y, dense, phi = qa_build(rows)
        return y, dense, phi, np.load(f"{CACHE}/asnq_ce_s_8000.npy"), "rate"
    if name == "triviaqa":
        rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy",
                                         allow_pickle=True)]
        y, dense, phi = qa_build(rows)
        return y, dense, phi, np.load(f"{CACHE}/triviaqa_ce_s.npy"), "rate"
    if name == "snli":
        from datasets import load_dataset
        d = load_dataset("stanfordnlp/snli", split="validation")
        rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
                for r in d if r["label"] in (0, 2)][:2400]
        hyp = [b for _, b, _ in rows]
        y = np.array([c for _, _, c in rows])
        Xh = TfidfVectorizer(max_features=8000,
                             ngram_range=(1, 2)).fit_transform(hyp)
        return y, Xh, Xh, np.load(f"{CACHE}/snli_s_2400.npy"), 0.5
    if name == "sick":
        rows = json.load(open("/tmp/sick_rows.json"))
        pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
                 for r in rows if r["label"] in (0, 2)][:2400]
        hyp = [b for _, b, _ in pairs]
        y = np.array([c for _, _, c in pairs])
        Xh = TfidfVectorizer(max_features=8000,
                             ngram_range=(1, 2)).fit_transform(hyp)
        return y, Xh, Xh, np.load(f"{CACHE}/sick_s.npy"), 0.5


def rf_probe(score, z):
    pr = cross_val_predict(RandomForestClassifier(200, random_state=0),
                           score.reshape(-1, 1), z, cv=CV,
                           method="predict_proba")[:, 1]
    return float(roc_auc_score(z, pr))


out = {}
for name in ["snli", "sick", "wikiqa", "asnq", "triviaqa"]:
    y, artX, phi, s, thr_rule = load(name)
    s = np.asarray(s, float)
    if thr_rule == "rate":
        art_model = make_pipeline(StandardScaler(with_mean=False),
                                  LogisticRegression(max_iter=1000))
    else:
        art_model = LogisticRegression(max_iter=1000)
    prob = cross_val_predict(art_model, artX, y, cv=CV,
                             method="predict_proba")[:, 1]
    index = cross_val_predict(art_model, artX, y, cv=CV,
                              method="decision_function")
    thr = (np.quantile(prob, 1 - y.mean()) if thr_rule == "rate" else 0.5)
    z = (prob > thr).astype(int)
    adv = (prob > thr) != (y == 1)

    srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)

    # conditional-quantile eraser on the channel index.
    # Equal-count bins via argsort split, and randomized ranks, so that s_q is
    # exactly Uniform(0,1) within every bin regardless of bin size or ties ---
    # otherwise bin-size/support differences leak (inverted) bin information.
    # bin edges aligned to the decision threshold: bin separately within z=0
    # and z=1 (proportional bin counts, min 1), so z is constant in every bin
    # and s_q is independent of z by construction.
    bins = np.empty(len(s), int)
    next_bin = 0
    for zv in (0, 1):
        grp = np.where(z == zv)[0]
        k = max(1, int(round(QBINS * len(grp) / len(s))))
        order = grp[np.argsort(index[grp], kind="stable")]
        for chunk in np.array_split(order, k):
            bins[chunk] = next_bin
            next_bin += 1
    sq = np.empty_like(s)
    for b in np.unique(bins):
        m = bins == b
        r = rankdata(s[m], method="ordinal") - rng.random(int(m.sum()))
        sq[m] = r / m.sum()

    # cross-fitted variant: bin edges and per-bin CDFs fit on train folds only,
    # applied to the held-out fold (mirrors the ridge operator's cross-fitting)
    sq_cf = np.empty_like(s)
    for tr, te in CV.split(s):
        for zv in (0, 1):
            trm = tr[z[tr] == zv]
            tem = te[z[te] == zv]
            if len(tem) == 0:
                continue
            if len(trm) < 10:
                sq_cf[tem] = rng.random(len(tem)); continue
            k = max(1, int(round(QBINS * len(trm) / len(tr))))
            edges = (np.quantile(index[trm], np.linspace(0, 1, k + 1)[1:-1])
                     if k > 1 else np.array([]))
            trb_id = np.digitize(index[trm], edges)
            teb_id = np.digitize(index[tem], edges)
            for b in range(k):
                trb = trm[trb_id == b]; teb = tem[teb_id == b]
                if len(teb) == 0:
                    continue
                if len(trb) < 3:
                    sq_cf[teb] = rng.random(len(teb)); continue
                srt = np.sort(s[trb])
                pos = np.searchsorted(srt, s[teb], side="left").astype(float)
                sq_cf[teb] = (pos + rng.random(len(teb))) / (len(trb) + 1)

    # permutation null floor: the same RF probe on a shuffled copy of sq ---
    # what "recoverability" the estimator reports on genuinely independent data
    null = rf_probe(rng.permutation(sq), z)

    A = lambda mask, u: float(roc_auc_score(y[mask], u[mask]))
    A_local = lambda u, yv: float(roc_auc_score(yv, u))
    A_slice_local = lambda u, yv, m: float(roc_auc_score(yv[m], u[m]))
    full = np.ones(len(y), bool)
    res = {"n": len(y),
           "rf_probe_z": {"raw": rf_probe(s, z), "linear": rf_probe(srep, z),
                          "quantile": rf_probe(sq, z)},
           "rf_probe_null_sym": float(max(null, 1 - null)),
           "rf_probe_z_sym_crossfit": float(max(rf_probe(sq_cf, z), 1 - rf_probe(sq_cf, z))),
           "A_full_crossfit": A_local(sq_cf, y),
           "A_slice_crossfit": A_slice_local(sq_cf, y, adv),
           "rf_probe_z_sym": {k: float(max(v, 1 - v)) for k, v in
                              {"raw": rf_probe(s, z), "linear": rf_probe(srep, z),
                               "quantile": rf_probe(sq, z)}.items()},
           "index_corr": {"raw": abs(float(np.corrcoef(s, index)[0, 1])),
                          "linear": abs(float(np.corrcoef(srep, index)[0, 1])),
                          "quantile": abs(float(np.corrcoef(sq, index)[0, 1]))},
           "A_full": {"raw": A(full, s), "linear": A(full, srep),
                      "quantile": A(full, sq)},
           "A_slice": {"raw": A(adv, s), "linear": A(adv, srep),
                       "quantile": A(adv, sq)}}
    out[name] = res
    print(name, json.dumps(res, indent=1))

json.dump(out, open(os.path.join(HERE, "..", "notes",
                                 "quantile_eraser.json"), "w"), indent=1)
print("saved")
