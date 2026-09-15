"""Problem-level bootstrap CIs for the bridge sweep's paired win rates.

Reconstructs the deterministic sweep of bridge_feasibility_probe.py from the same
caches (no re-scoring), collects per-problem win indicators for the raw and the
residualized score, and bootstraps 1,000 resamples over problems. Point values are
asserted against bridge_scorer_sweep.json / bridge_feasibility_probe.json before
any CI is trusted.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, HERE)
from bridge_feasibility_probe import build  # noqa: E402


def per_problem_wins(y, u, groups):
    wins, gids = [], []
    for gp in np.unique(groups):
        m = groups == gp
        if len(np.unique(y[m])) != 2:
            continue
        pos, neg = u[m][y[m] == 1][0], u[m][y[m] == 0][0]
        wins.append(1.0 if pos > neg else (0.5 if pos == neg else 0.0))
        gids.append(gp)
    return np.array(wins), np.array(gids)


def sweep_cis(scorer, nboot=1000):
    phi, y, verbose, pid, s = build(scorer)
    problems = np.unique(pid)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(problems))
    out = []
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        n_flip = int(round(p * len(problems)))
        flip = set(problems[order[:n_flip]].tolist())
        keep = [i for i in range(len(y))
                if verbose[i] == int((y[i] == 1) if pid[i] in flip else (y[i] == 0))]
        keep = np.array(keep)
        yk, sk, gk = y[keep], s[keep], pid[keep]
        cv = GroupKFold(min(5, len(np.unique(gk))))
        g = cross_val_predict(RidgeCV(alphas=[0.1, 1, 10]), phi[keep], sk,
                              cv=cv, groups=gk)
        srep = sk - g
        w_raw, _ = per_problem_wins(yk, sk, gk)
        w_rep, _ = per_problem_wins(yk, srep, gk)
        assert len(w_raw) == len(w_rep)
        brng = np.random.default_rng(1)
        idx = brng.integers(0, len(w_raw), size=(nboot, len(w_raw)))
        d = (w_rep - w_raw)[idx].mean(axis=1)
        r = w_raw[idx].mean(axis=1)
        out.append(dict(
            p=p, pairs=len(w_raw),
            win_raw=round(float(w_raw.mean()), 3),
            win_rep=round(float(w_rep.mean()), 3),
            win_delta=round(float((w_rep - w_raw).mean()), 3),
            win_delta_ci=[round(float(np.quantile(d, q)), 3) for q in (0.025, 0.975)],
            win_raw_ci=[round(float(np.quantile(r, q)), 3) for q in (0.025, 0.975)],
        ))
    return out


if __name__ == "__main__":
    ref = json.load(open(os.path.join(ROOT, "bridge_scorer_sweep.json")))
    result = {}
    for sc in ["rm", "haiku45", "sonnet5", "opus5"]:
        rows = sweep_cis(sc)
        for row in rows:
            r0 = [x for x in ref[sc] if x["p"] == row["p"]][0]
            for k in ("win_raw", "win_rep", "win_delta"):
                if abs(row[k] - r0[k]) > 0.0005:
                    raise SystemExit(f"MISMATCH {sc} p={row['p']} {k}: "
                                     f"{row[k]} vs {r0[k]}")
        result[sc] = rows
        for row in rows:
            print(sc, row["p"], row["win_delta"], row["win_delta_ci"])
    json.dump(result, open(os.path.join(ROOT, "bridge_win_cis.json"), "w"), indent=1)
    print("all point values reproduce the shipped sweep; CIs written")
