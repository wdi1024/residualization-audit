"""Feasibility probe for the entangled-regime bridge experiment.

The paper states (Section 7) that the obvious bridge design is blocked: making
the injected artifact correlated with the construct appears to require
label-dependent edits, and those cannot be certified semantics-preserving.

This probe tests a way around that. The edits stay exactly as they are --
comment/docstring-only rewrites, already verified by unit-test execution, and
produced without reference to the label. What changes is only *which* of the
four already-built cells per problem enters the evaluation sample. Keeping
(correct-verbose, buggy-terse) for a fraction p of problems and
(correct-terse, buggy-verbose) for the rest makes the format channel correlate
with correctness, with p controlling the strength. Sweeping p therefore sweeps
the construct-artifact collinearity on real text, a real scorer, and mechanical
labels.

This is a feasibility check, not a paper result: the collinearity is induced by
sampling rather than observed in the wild, so the setting is semi-synthetic.
CPU-only, uses the released MBPP cache.
"""

import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")


def format_features(code):
    """Content-blind format features: the channel the RM audit manipulates."""
    lines = code.split("\n")
    comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
    has_doc = 1.0 if re.search(r'"""|\'\'\'', code) else 0.0
    return [
        len(code),
        len(lines),
        comment_lines,
        has_doc,
        comment_lines / max(1, len(lines)),
        np.mean([len(ln) for ln in lines]) if lines else 0.0,
    ]


def build(scorer="rm"):
    rows = json.load(open(os.path.join(CACHE, "mbpp_rows.json")))
    fname = "mbpp_rm_s.npy" if scorer == "rm" else f"mbpp_judge_{scorer}_s.npy"
    s = np.load(os.path.join(CACHE, fname)).astype(float)
    prompts = [r[0] for r in rows]
    codes = [r[1] for r in rows]
    y = np.array([r[2] for r in rows])
    # field 3 is the problem index; the format arm is encoded in the cell label
    pid = np.array([r[3] for r in rows])
    verbose = np.array([1 if "verbose" in r[4] else 0 for r in rows])
    phi = np.array([format_features(c) for c in codes], float)
    return phi, y, verbose, pid, s


def audit(phi, y, s, groups):
    """One pass of the procedure's measurable quantities on a given sample."""
    cv = GroupKFold(min(5, len(np.unique(groups))))
    ahat = cross_val_predict(LogisticRegression(max_iter=2000), phi, y,
                             cv=cv, groups=groups, method="predict_proba")[:, 1]
    g = cross_val_predict(RidgeCV(alphas=[0.1, 1, 10]), phi, s, cv=cv, groups=groups)
    srep = s - g
    c1 = roc_auc_score(y, ahat)
    # R^2 of phi -> s, cross-fitted
    r2 = max(0.0, 1 - np.var(s - g) / np.var(s))
    thr = np.quantile(ahat, 1 - y.mean())
    adv = (ahat > thr) != (y == 1)
    out = {
        "n": int(len(y)),
        "C1": round(float(c1), 3),
        "R2_phi_s": round(float(r2), 3),
        "A_full_raw": round(float(roc_auc_score(y, s)), 3),
        "A_full_rep": round(float(roc_auc_score(y, srep)), 3),
        "n_slice": int(adv.sum()),
    }
    if 0 < adv.sum() < len(y) and len(np.unique(y[adv])) == 2:
        out["A_slice_raw"] = round(float(roc_auc_score(y[adv], s[adv])), 3)
        out["A_slice_rep"] = round(float(roc_auc_score(y[adv], srep[adv])), 3)
    out["gain_full"] = round(out["A_full_rep"] - out["A_full_raw"], 3)
    if "A_slice_raw" in out:
        out["gain_slice"] = round(out["A_slice_rep"] - out["A_slice_raw"], 3)

    # Within-problem paired win rate. Pooled AUC is the wrong instrument here:
    # the reward scale varies enormously across problems, which is why the paper
    # reports paired effects for this audit. Each problem contributes exactly one
    # correct and one buggy candidate, so the pairing is exact.
    # Ties count as half a win. The reward model emits floats and never ties,
    # but LLM judges emit integers on a 0-100 scale and tie on up to a fifth of
    # the pairs; scoring those as losses would understate every judge uniformly.
    def win_rate(u):
        wins = []
        for gp in np.unique(groups):
            m = groups == gp
            if len(np.unique(y[m])) != 2:
                continue
            pos, neg = u[m][y[m] == 1][0], u[m][y[m] == 0][0]
            wins.append(1.0 if pos > neg else (0.5 if pos == neg else 0.0))
        return float(np.mean(wins)), len(wins)

    wr_raw, npairs = win_rate(s)
    wr_rep, _ = win_rate(srep)
    out["pairs"] = npairs
    out["win_raw"] = round(wr_raw, 3)
    out["win_rep"] = round(wr_rep, 3)
    out["win_delta"] = round(wr_rep - wr_raw, 3)
    return out


def sweep(scorer="rm"):
    phi, y, verbose, pid, s = build(scorer)
    problems = np.unique(pid)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(problems))
    results = []
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        # deterministic split of problems: the first p share get the
        # (correct-verbose, buggy-terse) pairing, the rest the mirror image.
        n_flip = int(round(p * len(problems)))
        flip = set(problems[order[:n_flip]].tolist())
        keep = []
        for i in range(len(y)):
            want_verbose = (y[i] == 1) if pid[i] in flip else (y[i] == 0)
            if verbose[i] == int(want_verbose):
                keep.append(i)
        keep = np.array(keep)
        rho = float(np.corrcoef(verbose[keep], y[keep])[0, 1])
        r = audit(phi[keep], y[keep], s[keep], pid[keep])
        r["p"] = p
        r["scorer"] = scorer
        r["corr_format_correctness"] = round(rho, 3)
        results.append(r)
    return results


if __name__ == "__main__":
    scorers = sys.argv[1:] or ["rm"]
    out = {}
    for sc in scorers:
        out[sc] = sweep(sc)
        for r in out[sc]:
            print(json.dumps(r))
    dest = ("bridge_feasibility_probe.json" if scorers == ["rm"]
            else "bridge_scorer_sweep.json")
    with open(os.path.join(CACHE, "..", dest), "w") as fh:
        json.dump(out["rm"] if scorers == ["rm"] else out, fh, indent=2)
