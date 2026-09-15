"""Placebo slice test (round-9 review, item 2): estimate the size of the
error-conditioning bias on real data.

The concern: Property 4 says conditioning on the surface predictor's errors
structurally favors a positive slice gain (proved only in an idealized model),
and the paper leans on that heuristic to down-weight the validation check. The
requested measurement: compare the declared-slice gain against gains on
PLACEBO slices --- slices defined by the errors of a synthetic predictor with
the SAME construct AUC as the surface predictor (matched C1) but errors
unrelated to phi. If generic error-conditioning at matched AUC produced the
gain, placebo gains would match the declared gain; if the gain is specific to
the surface channel (the attribution claim), placebo gains should sit near the
full-population gain instead.

Placebo construction: u_i = delta*y_i + N(0,1) with delta = sqrt(2)*Phi^{-1}(C1),
so AUC(y, u) ~= C1 in expectation; the slice rule is the paper's rate-calibrated
rule applied to u verbatim. B = 200 draws, seed 0.

Settings: the three QA settings sharing the paper's recipe --- SQuAD (held-out
anchor), WikiQA (screened positive), DuoRC (held-out #2). Builders verbatim
from heldout_squad_confirmatory.py / wikiqa_grouped.py /
heldout_duorc_confirmatory.py; scores from the existing caches (no new
scoring). Pipeline = deployment protocol (question-grouped GroupKFold(5)),
ridge alpha=1, rate-calibrated slice.

Committed prediction (written before computing placebo gains): placebo-slice
gains concentrate near the full-population gain (negative), far below the
declared-slice gain, in all three settings.

Determinism: run, then --confirm in a separate process; exact equality of all
recorded statistics required.

Usage: python3 placebo_slice_test.py [--confirm]
Writes ../notes/placebo_slice_test.json (+ .committed).
"""
import json
import os
import re
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "..", "notes")
CACHE = os.path.join(HERE, "..", "cache")
B = 200


def overlap(a, b):
    A, Bs = set(a.lower().split()), set(b.lower().split())
    return len(A & Bs) / max(1, len(A | Bs))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def build_squad():
    from datasets import load_dataset
    d = load_dataset("squad", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1:
            continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0))
            qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000:
            break
    rows = rows[:8000]
    return rows, np.array(qid[:8000]), "squad_ce_s_8000.npy"


def build_duorc():
    from datasets import load_dataset
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        if r["no_answer"] or not r["answers"]:
            continue
        sents = sent_split(r["plot"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"][0]
        if len(ans.split()) > 12 or not ans.strip():
            continue
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1:
            continue
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0))
            qid.append(seen_q)
        seen_q += 1
        if len(rows) >= 8000:
            break
    rows = rows[:8000]
    return rows, np.array(qid[:8000]), "duorc_ce_s_8000.npy"


def build_wikiqa():
    from datasets import load_dataset
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(a for a, _, _ in rows))}
    qid = np.array([uniq[a] for a, _, _ in rows])
    return rows, qid, "wikiqa_ce_s_8000.npy"


BUILDERS = {"squad": build_squad, "wikiqa": build_wikiqa, "duorc": build_duorc}


def run_setting(name):
    rows, groups, cachef = BUILDERS[name]()
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load(os.path.join(CACHE, cachef))
    assert len(s) == len(y), f"{name}: cache length mismatch"

    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    phi = hstack([TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
                  .fit_transform(ans), csr_matrix(dense)]).tocsr()

    cv_kw = dict(cv=GroupKFold(5), groups=groups)
    art_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    prob = cross_val_predict(art_model, dense, y, method="predict_proba",
                             **cv_kw)[:, 1]
    c1 = float(roc_auc_score(y, prob))
    thr = float(np.quantile(prob, 1 - y.mean()))
    adv = (prob > thr) != (y == 1)
    s_rep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, **cv_kw)

    def A(mask, u):
        return float(roc_auc_score(y[mask], u[mask]))
    full = np.ones(len(y), bool)
    g_obs = A(adv, s_rep) - A(adv, s)
    g_full = A(full, s_rep) - A(full, s)

    # placebo slices: matched-AUC synthetic predictor, errors unrelated to phi
    delta = float(np.sqrt(2) * norm.ppf(c1))
    rng = np.random.default_rng(0)
    gains, aucs, sizes = [], [], []
    for _ in range(B):
        u = delta * y + rng.standard_normal(len(y))
        thr_u = np.quantile(u, 1 - y.mean())
        adv_u = (u > thr_u) != (y == 1)
        if y[adv_u].min() == y[adv_u].max():
            continue
        gains.append(A(adv_u, s_rep) - A(adv_u, s))
        aucs.append(float(roc_auc_score(y, u)))
        sizes.append(int(adv_u.sum()))
    gains = np.array(gains)
    return {
        "setting": name, "n": len(y), "pos_rate": float(y.mean()),
        "C1_grouped": c1, "n_slice_declared": int(adv.sum()),
        "declared_slice_gain": g_obs, "full_population_gain": g_full,
        "placebo_B": len(gains),
        "placebo_gain_mean": float(gains.mean()),
        "placebo_gain_p2.5": float(np.percentile(gains, 2.5)),
        "placebo_gain_p97.5": float(np.percentile(gains, 97.5)),
        "placebo_gain_max": float(gains.max()),
        "placebo_auc_mean": float(np.mean(aucs)),
        "placebo_slice_size_mean": float(np.mean(sizes)),
        "excess_over_placebo": float(g_obs - gains.mean()),
    }


def main(confirm=False):
    t0 = time.time()
    cfile = os.path.join(NOTES, "placebo_slice_test.json.committed")
    committed = {
        "experiment": "placebo slice test (matched-AUC synthetic predictor)",
        "settings": ["squad", "wikiqa", "duorc"],
        "placebo": "u = sqrt(2)*Phi^-1(C1)*y + N(0,1), rate-calibrated slice "
                   "rule verbatim, B=200, seed 0",
        "protocol": "question-grouped GroupKFold(5), ridge alpha=1, "
                    "cached scorer outputs (no new scoring)",
        "committed_prediction": "placebo-slice gains concentrate near the "
                                "full-population gain (negative), far below "
                                "the declared-slice gain, in all three "
                                "settings",
    }
    if not os.path.exists(cfile):
        json.dump(committed, open(cfile, "w"), indent=1)
        print("COMMITTED BEFORE COMPUTING", flush=True)

    results = [run_setting(nm) for nm in ["squad", "wikiqa", "duorc"]]
    for r in results:
        print(f"[{r['setting']}] C1={r['C1_grouped']:.3f} "
              f"declared={r['declared_slice_gain']:+.3f} "
              f"placebo={r['placebo_gain_mean']:+.3f} "
              f"[{r['placebo_gain_p2.5']:+.3f},{r['placebo_gain_p97.5']:+.3f}] "
              f"full={r['full_population_gain']:+.3f}", flush=True)

    rfile = os.path.join(NOTES, "placebo_slice_test.json")
    out = {"results": results, "committed": committed,
           "runtime_s": round(time.time() - t0, 1)}
    if confirm:
        prev = json.load(open(rfile))
        assert json.dumps(prev["results"], sort_keys=True) == \
            json.dumps(results, sort_keys=True), "determinism FAIL"
        prev["determinism_check"] = ("PASS: all statistics identical across "
                                     "two processes")
        json.dump(prev, open(rfile, "w"), indent=1)
        print("determinism CONFIRMED, stamped", rfile)
    else:
        json.dump(out, open(rfile, "w"), indent=1)
        print("wrote", rfile)


if __name__ == "__main__":
    main(confirm="--confirm" in sys.argv)
