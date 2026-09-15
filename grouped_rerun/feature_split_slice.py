"""Bound the Property 4 conditioning bias by splitting phi.

Property 4 says the declared slice favors subtraction because the slice is the set
where a phi-fit predictor errs and the subtracted component is fit on the same phi.
The mechanism needs the two sides to share features. This script removes the sharing:
the slice is defined by a predictor fit on block A, the repair residualizes s on a
block B disjoint from A, and the gain is measured on that slice.

Three splits, each with A (slice) and B (repair) disjoint:

  S1  A = answer TF-IDF          B = dense (overlap, lengths)
  S2  A = lengths                B = TF-IDF + overlap
  S3  A = overlap                B = TF-IDF + lengths

Everything else is the paper's recipe unchanged: GroupKFold(5) on question id, ridge
alpha=1, the same slice rule (top-(1-base rate) quantile of the predictor, then the
rows where that thresholded prediction disagrees with y), query-clustered bootstrap.

Also reported per split:
  c1_A     the slice predictor's AUC, so an uninformative slice is visible as such
  corr_AB  |corr| between the A-predictor's score and the fitted B-component g_hat,
           which is what is left of the shared-information route after the split
  baseline the same setting's full-phi gain under the paper's own slice, for reference
"""
import json, os, re
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
rng = np.random.default_rng(0)
NBOOT = 1000


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def load_wikiqa():
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    q = [a for a, _, _ in rows]; ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"), np.array([uniq[qq] for qq in q]), None


def load_asnq():
    rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
    q = [a for a, _, _ in rows]; ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/asnq_ce_s_8000.npy"), np.array([uniq[qq] for qq in q]), None


def load_trivia():
    rows = np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)
    q = [r[0] for r in rows]; ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/triviaqa_ce_s.npy"), np.array([uniq[qq] for qq in q]), None


def load_squad():
    d = load_dataset("squad", split="validation")
    rows, qid, y2_list, seen_q = [], [], [], 0
    for r in d:
        sents = sent_split(r["context"])
        if not (3 <= len(sents) <= 25):
            continue
        ans = r["answers"]["text"][0]
        pos = [i for i, s in enumerate(sents) if ans in s]
        if len(pos) != 1:
            continue
        others = r["answers"]["text"][1:]
        for i, s in enumerate(sents):
            rows.append((r["question"], s, 1 if i == pos[0] else 0))
            qid.append(seen_q)
            y2_list.append(1 if (others and any(o in s for o in others)) else (0 if others else -1))
        seen_q += 1
        if len(rows) >= 8000:
            break
    rows = rows[:8000]
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows]), np.load(f"{CACHE}/squad_ce_s_8000.npy"),
            np.array(qid[:8000]), np.array(y2_list[:8000]))


def q_boot_gain(y, raw, rep, mask, qid, nboot=NBOOT):
    qs = np.unique(qid[mask])
    by_q = {qq: np.where(mask & (qid == qq))[0] for qq in qs}
    gains = []
    for _ in range(nboot):
        pick = rng.choice(qs, size=len(qs), replace=True)
        idx = np.concatenate([by_q[qq] for qq in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        gains.append(roc_auc_score(yy, rep[idx]) - roc_auc_score(yy, raw[idx]))
    return [round(float(np.percentile(gains, 2.5)), 4), round(float(np.percentile(gains, 97.5)), 4)]


def slice_from(block, y, qid, cv, sparse):
    """The paper's slice rule, applied to whatever feature block is passed in."""
    if sparse:
        est = LogisticRegression(max_iter=1000)
    else:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(est, block, y, cv=cv, groups=qid, method="predict_proba")[:, 1]
    thr = np.quantile(p, 1 - y.mean())
    z = p > thr
    return p, (z != (y == 1))


def run(name, q, ans, y, s, qid, y2):
    print(f"\n=== {name}: n={len(y)} pos={y.mean():.3f} queries={len(np.unique(qid))}", flush=True)
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    lens = np.hstack([la, lq, la - lq])
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    cv = GroupKFold(5)

    # the paper's own configuration, for reference on the same rows
    _, adv0 = slice_from(dense, y, qid, cv, sparse=False)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    g0 = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, groups=qid)
    rep0 = s - g0
    out = {"n": int(len(y)), "n_adv_paper": int(adv0.sum()),
           "paper_slice_gain": round(float(roc_auc_score(y[adv0], rep0[adv0])
                                           - roc_auc_score(y[adv0], s[adv0])), 4)}

    splits = {
        "S1_tfidf_slice_dense_repair":   (Xa, True, csr_matrix(dense).tocsr()),
        "S2_length_slice_lexoverlap_repair": (lens, False, hstack([Xa, csr_matrix(ov)]).tocsr()),
        "S3_overlap_slice_lexlength_repair": (ov, False, hstack([Xa, csr_matrix(lens)]).tocsr()),
    }

    out["splits"] = {}
    for tag, (A, A_sparse, B) in splits.items():
        pA, advB = slice_from(A, y, qid, cv, sparse=A_sparse)
        gB = cross_val_predict(Ridge(alpha=1.0), B, s, cv=cv, groups=qid)
        repB = s - gB
        r2B = 1 - float(((s - gB) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
        rec = {
            "c1_A": round(float(roc_auc_score(y, pA)), 4),
            "R2_B": round(r2B, 4),
            "n_adv": int(advB.sum()),
            "corr_AB": round(abs(float(np.corrcoef(pA, gB)[0, 1])), 4),
            "slice_overlap_with_paper": round(float((advB & adv0).sum() / max(1, advB.sum())), 4),
        }
        if advB.sum() > 20 and y[advB].min() != y[advB].max():
            a_raw = float(roc_auc_score(y[advB], s[advB]))
            a_rep = float(roc_auc_score(y[advB], repB[advB]))
            rec.update({"adv": [round(a_raw, 4), round(a_rep, 4)],
                        "gain": round(a_rep - a_raw, 4),
                        "gain_ci": q_boot_gain(y, s, repB, advB, qid)})
            full = np.ones(len(y), bool)
            rec["full"] = [round(float(roc_auc_score(y, s)), 4),
                           round(float(roc_auc_score(y, repB)), 4)]
            # SQuAD only: the disjoint annotator reading, on the split slice
            if y2 is not None:
                m2 = (y2 >= 0) & advB
                if m2.sum() > 20 and y2[m2].min() != y2[m2].max():
                    rec["adv_y2"] = [round(float(roc_auc_score(y2[m2], s[m2])), 4),
                                     round(float(roc_auc_score(y2[m2], repB[m2])), 4)]
                    rec["gain_ci_y2"] = q_boot_gain(y2, s, repB, m2, qid)
        else:
            rec["gain"] = None
        out["splits"][tag] = rec
        print(f"  {tag}: {json.dumps(rec)}", flush=True)
    return out


if __name__ == "__main__":
    results = {}
    for nm, loader in [("wikiqa", load_wikiqa), ("asnq", load_asnq),
                       ("triviaqa", load_trivia), ("squad", load_squad)]:
        try:
            results[nm] = run(nm, *loader())
        except Exception as e:
            print(f"!! {nm} failed: {type(e).__name__}: {e}", flush=True)
            results[nm] = {"error": f"{type(e).__name__}: {e}"}

    with open(os.path.join(HERE, "feature_split_slice_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nwritten feature_split_slice_results.json")
