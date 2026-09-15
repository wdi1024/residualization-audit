"""ASNQ candidate-protocol sensitivity sweep (mock-review remaining item #2).

The reviewer's devil's-advocate concern: ASNQ's natural row sampling gave C1=0.643
(gate fail); the paper's resampled candidate protocol (all-pos + <=19 neg/question,
WikiQA-comparable) gave C1=0.747 (pass). Since C1 was observed before the protocol
choice, partial adaptivity remains. Required sensitivity analysis:
  * candidates per query: 5, 10, 20, 50, all
  * several negative-sampling seeds
  * natural vs balanced base rate
  * joint movement of C1, C2, and repair gain

Design: for each config, rebuild the candidate lists from ASNQ validation
(questions with >=1 positive, dataset order shuffled by rng(seed); all positives +
up to k_neg sampled negatives per question; accumulate to 8000 rows, truncate —
mirroring the screen's row cap). Extra configs: "all" (every candidate, natural
base rate within pos-questions), "balanced" (1:1 neg:pos per question), "natural"
(screen-verbatim original protocol incl. no-positive questions, rng(0)), and
"cached20" (the EXACT July resampled 8k rows from cache/asnq_rows_8000.json with
their cached CE scores — exact 0.747 anchor).

Recipe verbatim from asnq_positive_run.py / grouped rerun protocol:
  phi = answer-only TF-IDF(8000, 1-2gram) + dense [overlap, len_a, len_q, la-lq];
  C1 screen = dense-only LogisticRegression, KFold(5, shuffle, seed 0) — the
  screen's own CV, comparable to the 0.643 / 0.747 anchors;
  pipeline = question-grouped GroupKFold(5) (deployment protocol, deterministic
  first-occurrence group ids — never hash()); repair = Ridge(alpha=1) residual;
  slice rate-calibrated; gates C1>=0.65, R2>=0.05, Dadv>=0.03, cert<=0.05;
  bootstrap 1000 question-clustered resamples.
Scorer: cross-encoder/ms-marco-MiniLM-L-6-v2 (max_len 256), global score cache
keyed by sha1(question \x1f sentence) so every (q, sent) pair is scored once
across the whole grid; seeded from the July cache.

Single-shot protocol: committed artifact (grid, gates, anchors, prediction)
written BEFORE any new scoring. Determinism: run, then --confirm in a separate
process (recompute all stats from the on-disk score cache + re-score a fixed
32-pair spot batch; exact equality required).

Usage: python3 asnq_candidate_sensitivity.py [--confirm]
Writes ../notes/asnq_candidate_sensitivity.json (+ .committed).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "..", "notes")
CACHE = os.path.join(HERE, "..", "cache")
SCREEN_CV = KFold(5, shuffle=True, random_state=0)
GATES = {"C1": 0.65, "C2_R2": 0.05, "C2_Dadv": 0.03, "cert": 0.05}
SCORER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAXLEN = 256
NCAP = 8000
SEEDS = [0, 1, 2]
KNEGS = [4, 9, 19, 49]          # -> candidates/query ~5, 10, 20, 50
SCORES_PATH = os.path.join(CACHE, "asnq_sweep_scores.json")


def key(q, a):
    return hashlib.sha1((q + "\x1f" + a).encode()).hexdigest()


def first_occurrence_ids(keys):
    seen = {}
    return np.array([seen.setdefault(k, len(seen)) for k in keys])


# ---------------- candidate-set builders ----------------
def load_asnq():
    from datasets import load_dataset
    d = load_dataset("asnq", split="validation")
    by_q = {}
    for r in d:
        by_q.setdefault(r["question"], []).append((r["sentence"], int(r["label"])))
    return by_q


def build_config(by_q, mode, k_neg=None, seed=0):
    """Returns rows [(q, sent, y)] capped at NCAP."""
    rng = np.random.default_rng(seed)
    if mode == "natural":               # screen-verbatim original protocol
        qs = list(by_q.keys())
        rng.shuffle(qs)
        rows = []
        for qq in qs:
            for sent, lab in by_q[qq]:
                rows.append((qq, sent, lab))
            if len(rows) >= NCAP:
                break
        return rows[:NCAP]
    qs = [qq for qq, cands in by_q.items() if any(l == 1 for _, l in cands)]
    rng.shuffle(qs)
    rows = []
    for qq in qs:
        pos = [(s, l) for s, l in by_q[qq] if l == 1]
        neg = [(s, l) for s, l in by_q[qq] if l == 0]
        if mode == "kneg":
            take = min(k_neg, len(neg))
        elif mode == "balanced":
            take = min(len(pos), len(neg))
        elif mode == "all":
            take = len(neg)
        if take < len(neg):
            idx = np.sort(rng.choice(len(neg), take, replace=False))
            neg = [neg[i] for i in idx]
        for s, l in pos + neg:
            rows.append((qq, s, l))
        if len(rows) >= NCAP:
            break
    return rows[:NCAP]


def config_grid(by_q):
    cfgs = {}
    for k in KNEGS:
        for sd in SEEDS:
            cfgs[f"k{k+1}_s{sd}"] = build_config(by_q, "kneg", k, sd)
    for sd in SEEDS:
        cfgs[f"all_s{sd}"] = build_config(by_q, "all", seed=sd)
        cfgs[f"balanced_s{sd}"] = build_config(by_q, "balanced", seed=sd)
    cfgs["natural"] = build_config(by_q, "natural", seed=0)
    rows20 = [tuple(r) for r in
              json.load(open(os.path.join(CACHE, "asnq_rows_8000.json")))]
    cfgs["cached20"] = [(a, b, int(c)) for a, b, c in rows20]
    return cfgs


# ---------------- scoring with global pair cache ----------------
def load_score_cache():
    sc = {}
    if os.path.exists(SCORES_PATH):
        sc = json.load(open(SCORES_PATH))
    else:  # seed from the July cache (exact resampled rows + CE scores)
        rows = json.load(open(os.path.join(CACHE, "asnq_rows_8000.json")))
        s = np.load(os.path.join(CACHE, "asnq_ce_s_8000.npy"))
        assert len(rows) == len(s)
        for (qq, aa, _), v in zip(rows, s):
            sc[key(qq, aa)] = float(v)
    return sc


def score_missing(cfgs, sc, confirm):
    todo, seen = [], set()
    for rows in cfgs.values():
        for qq, aa, _ in rows:
            k = key(qq, aa)
            if k not in sc and k not in seen:
                seen.add(k)
                todo.append((qq, aa, k))
    spot = {}
    if confirm:
        assert not todo, f"--confirm but {len(todo)} pairs unscored"
        # fixed spot batch: first 32 rows of cached20 (deterministic)
        rows = cfgs["cached20"][:32]
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(SCORER, max_length=MAXLEN)
        sa = np.asarray(ce.predict([(qq, aa) for qq, aa, _ in rows],
                                   batch_size=64), float)
        ref = np.array([sc[key(qq, aa)] for qq, aa, _ in rows])
        spot = {"spot_check_n": 32,
                "spot_check_max_absdiff": float(np.abs(sa - ref).max())}
        assert spot["spot_check_max_absdiff"] < 5e-4, \
            f"scorer nondeterminism: {spot}"
        return sc, spot
    if todo:
        print(f"scoring {len(todo)} new (q, sent) pairs", flush=True)
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(SCORER, max_length=MAXLEN)
        B = 2048
        for b in range(0, len(todo), B):
            chunk = todo[b:b + B]
            s = np.asarray(ce.predict([(qq, aa) for qq, aa, _ in chunk],
                                      batch_size=64), float)
            for (_, _, k), v in zip(chunk, s):
                sc[k] = float(v)
            print(f"  scored {min(b + B, len(todo))}/{len(todo)}", flush=True)
        json.dump(sc, open(SCORES_PATH, "w"))
    return sc, spot


# ---------------- per-config pipeline (recipe verbatim) ----------------
def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def run_config(name, rows, sc):
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    n = len(y)
    s = np.array([sc[key(a, b)] for a, b, _ in rows])
    groups = first_occurrence_ids(q)
    n_q = len(np.unique(groups))

    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    phi = hstack([TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
                  .fit_transform(ans), csr_matrix(dense)]).tocsr()

    def c1_model():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    out = {"config": name, "n": n, "n_questions": n_q,
           "pos_rate": float(y.mean())}
    if y.min() == y.max():
        out["verdict"] = "degenerate labels"
        return out

    # C1, screen CV verbatim (comparable to 0.643 / 0.747 anchors)
    p = cross_val_predict(c1_model(), dense, y, cv=SCREEN_CV,
                          method="predict_proba")[:, 1]
    out["C1_screen_cv"] = float(roc_auc_score(y, p))

    # pipeline: question-grouped (deployment protocol)
    grouped = (1 - n_q / n) >= 0.05
    cv_kw = dict(cv=GroupKFold(5), groups=groups) if grouped \
        else dict(cv=SCREEN_CV)
    art = cross_val_predict(c1_model(), dense, y, method="predict_proba",
                            **cv_kw)[:, 1]
    index = cross_val_predict(c1_model(), dense, y, method="decision_function",
                              **cv_kw)
    out["grouped_cv"] = grouped
    out["C1_grouped"] = float(roc_auc_score(y, art))
    thr = float(np.quantile(art, 1 - y.mean()))
    adv = (art > thr) != (y == 1)
    full = np.ones(n, bool)
    pred = cross_val_predict(Ridge(alpha=1.0), phi, s, **cv_kw)
    srep = s - pred
    r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())

    def A(m, u):
        return float(roc_auc_score(y[m], u[m])) if y[m].min() != y[m].max() \
            else None
    dadv = (A(full, s) - A(adv, s)) if A(adv, s) is not None else None
    out.update({"C2_R2_phi_to_s": r2, "C2_Delta_adv": dadv,
                "A_full": [A(full, s), A(full, srep)],
                "A_adv": [A(adv, s), A(adv, srep)], "n_adv": int(adv.sum())})

    rng = np.random.default_rng(0)
    gidx = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    gs = np.array(sorted(gidx))

    def gain_ci(mask, B=1000):
        gains = []
        for _ in range(B):
            take = rng.choice(gs, len(gs), replace=True)
            idx = np.concatenate([gidx[t] for t in take])
            m = mask[idx]
            ya, sa, ra = y[idx][m], s[idx][m], srep[idx][m]
            if len(ya) == 0 or ya.min() == ya.max():
                continue
            gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
        if not gains:
            return None
        return [float(np.percentile(gains, 2.5)),
                float(np.percentile(gains, 97.5)), len(gains)]

    slice_ci = gain_ci(adv)
    full_ci = gain_ci(full)
    out["slice_gain"] = (A(adv, srep) - A(adv, s)) \
        if A(adv, s) is not None else None
    out["slice_gain_CI"] = slice_ci
    out["full_gain"] = A(full, srep) - A(full, s)
    out["full_gain_CI"] = full_ci
    out["decorrelation_before_after"] = [
        abs(float(np.corrcoef(s, index)[0, 1])),
        abs(float(np.corrcoef(srep, index)[0, 1]))]

    c1_pass = out["C1_screen_cv"] >= GATES["C1"]
    c2_pass = r2 >= GATES["C2_R2"] and dadv is not None \
        and dadv >= GATES["C2_Dadv"]
    cert_pass = out["decorrelation_before_after"][1] <= GATES["cert"]
    sig_pos = slice_ci is not None and slice_ci[0] > 0
    out["gates"] = {"C1": bool(c1_pass), "C2": bool(c2_pass),
                    "cert": bool(cert_pass), "slice_sig_pos": bool(sig_pos)}
    out["verdict"] = ("validated positive" if c1_pass and c2_pass and cert_pass
                      and sig_pos else
                      "screened out (C1)" if not c1_pass else "not validated")
    return out


def main(confirm=False):
    t0 = time.time()
    by_q = load_asnq()
    cfgs = config_grid(by_q)

    cfile = os.path.join(NOTES, "asnq_candidate_sensitivity.json.committed")
    committed = {
        "experiment": "ASNQ candidate-protocol sensitivity sweep",
        "grid": {nm: {"n": len(r),
                      "pos_rate": float(np.mean([c for _, _, c in r]))}
                 for nm, r in cfgs.items()},
        "scorer": SCORER, "gates": GATES,
        "anchors": {"natural_screen_C1": 0.643, "resampled_screen_C1": 0.747},
        "recipe": "asnq_positive_run.py verbatim; C1 screen KFold(5,s,0) on "
                  "dense; pipeline question-grouped GroupKFold(5); Ridge(1) "
                  "residual; rate-calibrated slice; 1000 question-clustered "
                  "bootstrap",
        "committed_prediction": "C1 declines monotonically toward 0.643 as "
                                "candidates/query grows; the sweep decides "
                                "whether the k~20 gate pass is knife-edge or a "
                                "plateau; seed variation expected small; C2 and "
                                "repair sign expected stable wherever the slice "
                                "is measurable",
    }
    if not os.path.exists(cfile):
        json.dump(committed, open(cfile, "w"), indent=1)
        print("COMMITTED BEFORE SCORING", flush=True)

    sc, spot = score_missing(cfgs, sc=load_score_cache(), confirm=confirm)

    results = []
    for nm, rows in cfgs.items():
        r = run_config(nm, rows, sc)
        results.append(r)
        print(f"[{nm}] n={r['n']} pos={r['pos_rate']:.3f} "
              f"C1={r.get('C1_screen_cv', float('nan')):.3f} "
              f"verdict={r['verdict']}", flush=True)

    out = {"results": results, "committed": committed,
           "runtime_s": round(time.time() - t0, 1)}
    rfile = os.path.join(NOTES, "asnq_candidate_sensitivity.json")
    if confirm:
        prev = json.load(open(rfile))
        a = json.dumps(prev["results"], sort_keys=True)
        b = json.dumps(results, sort_keys=True)
        assert a == b, "determinism FAIL: statistics differ across processes"
        prev.update(spot)
        prev["determinism_check"] = ("PASS: all statistics identical across two "
                                     "processes; scorer spot-check max|diff|="
                                     f"{spot['spot_check_max_absdiff']}")
        json.dump(prev, open(rfile, "w"), indent=1)
        print("determinism CONFIRMED, stamped", rfile)
    else:
        json.dump(out, open(rfile, "w"), indent=1)
        print("wrote", rfile)


if __name__ == "__main__":
    main(confirm="--confirm" in sys.argv)
