"""Bootstrap confidence intervals for the certificate on every positive.

Section 3.3 states the certificate as a point criterion (R_lin <= 0.05) but the
held-out runs report bootstrap CIs, so the two held-out settings were judged on a
stricter rule than the rest. This script closes that gap: it recomputes
R_lin(s_rep) = |corr(s_rep, c_hat)| for the five positives with the paper's own
pipeline and attaches a 95% bootstrap CI to each, clustered by question wherever
the evaluation unit is a per-question candidate list.

Loaders and the analysis recipe are reused verbatim from
review_r3_experiments.py: its source is executed up to (but not including) its
main block, so the artifact model, folds, and features are identical.
"""

import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "review_r3_experiments.py")
CACHE = os.path.join(HERE, "..", "cache")
N_BOOT = 1000
GATE = 0.05

# reuse the exact loaders/feature builders without running that script's main
src = open(SRC).read()
prefix = src.split("\nout = {}", 1)[0]
ns: dict = {"__file__": SRC}
exec(compile(prefix, SRC, "exec"), ns)


def rlin(srep, index):
    return abs(float(np.corrcoef(srep, index)[0, 1]))


def boot_ci(srep, index, clusters=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(srep)
    if clusters is None:
        idx_pool = [np.array([i]) for i in range(n)]
    else:
        groups: dict = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        idx_pool = [np.array(v) for v in groups.values()]
    k = len(idx_pool)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k, size=k)
        sel = np.concatenate([idx_pool[j] for j in pick])
        vals.append(rlin(srep[sel], index[sel]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(name, y, artX, phi, s, standardize_art, clusters=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s = np.asarray(s, float)
    art_model = (make_pipeline(StandardScaler(with_mean=False),
                               LogisticRegression(max_iter=1000))
                 if standardize_art else LogisticRegression(max_iter=1000))
    index = cross_val_predict(art_model, artX, y, cv=ns["CV"],
                              method="decision_function")
    srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=ns["CV"])
    pt = rlin(srep, index)
    lo, hi = boot_ci(srep, index, clusters)
    verdict = "clean" if hi <= GATE else ("marginal" if pt <= GATE else "fail")
    print(f"{name:>10}  R_lin(s_rep) = {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"({'clustered' if clusters is not None else 'row'})  -> {verdict}")
    return {"setting": name, "r_lin_rep": pt, "ci_lo": lo, "ci_hi": hi,
            "clustered": clusters is not None, "verdict": verdict}


out = []

y, artX, phi, s, _ = ns["load_snli"]()
out.append(run("SNLI", y, artX, phi, s, standardize_art=False))

y, artX, phi, s, _ = ns["load_sick"]()
out.append(run("SICK", y, artX, phi, s, standardize_art=False))

from datasets import load_dataset  # noqa: E402

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
y, dense, phi = ns["qa_build"](rows)
s = np.load(f"{CACHE}/wikiqa_ce_s_8000.npy")
out.append(run("WikiQA", y, dense, phi, s, True, [r[0] for r in rows]))

rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
y, dense, phi = ns["qa_build"](rows)
s = np.load(f"{CACHE}/asnq_ce_s_8000.npy")
out.append(run("ASNQ", y, dense, phi, s, True, [r[0] for r in rows]))

rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)]
y, dense, phi = ns["qa_build"](rows)
s = np.load(f"{CACHE}/triviaqa_ce_s.npy")
out.append(run("TriviaQA", y, dense, phi, s, True, [r[0] for r in rows]))

path = os.path.join(HERE, "..", "notes", "cert_ci.json")
json.dump(out, open(path, "w"), indent=1)
print("\nsaved", path)
