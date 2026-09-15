"""Disambiguate the feature-split result: is the loss from breaking the coupling,
or from repairing with a weaker feature block?

feature_split_slice.py made the slice block A and the repair block B disjoint, and the
gain went null or negative in every cell where all three gates passed. Two readings:

  (i)  Property 4's coupling was carrying the observational gain, or
  (ii) B is a strict subset of phi, so the repair simply removes less.

This separates them. The slice is defined by block A exactly as before, but the repair
uses the FULL phi, as in the paper. The subtracted component is then identical to the
paper's; only the conditioning changes. If the gain returns here, (ii) explains the
earlier result. If it stays null or negative, (i) does.

Everything else is the paper's recipe: GroupKFold(5) on question id, ridge alpha=1,
same slice rule, query-clustered bootstrap.
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

from feature_split_slice import (load_wikiqa, load_asnq, load_trivia, load_squad,
                                 overlap, q_boot_gain, slice_from)

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, q, ans, y, s, qid, y2):
    print(f"\n=== {name}: n={len(y)}", flush=True)
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    lens = np.hstack([la, lq, la - lq])
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    cv = GroupKFold(5)

    # the paper's subtracted component, unchanged
    g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, groups=qid)
    rep = s - g
    full_raw = float(roc_auc_score(y, s))

    out = {"R2_phi": round(1 - float(((s - g) ** 2).sum())
                           / float(((s - s.mean()) ** 2).sum()), 4),
           "full": [round(full_raw, 4), round(float(roc_auc_score(y, rep)), 4)],
           "slices": {}}

    blocks = {"paper_dense": (dense, False),
              "A_tfidf": (Xa, True),
              "A_length": (lens, False),
              "A_overlap": (ov, False)}

    for tag, (A, sp) in blocks.items():
        pA, adv = slice_from(A, y, qid, cv, sparse=sp)
        if adv.sum() < 20 or y[adv].min() == y[adv].max():
            out["slices"][tag] = {"gain": None}
            continue
        a_raw = float(roc_auc_score(y[adv], s[adv]))
        a_rep = float(roc_auc_score(y[adv], rep[adv]))
        rec = {"c1_A": round(float(roc_auc_score(y, pA)), 4),
               "n_adv": int(adv.sum()),
               "delta_adv": round(full_raw - a_raw, 4),
               "adv": [round(a_raw, 4), round(a_rep, 4)],
               "gain": round(a_rep - a_raw, 4),
               "gain_ci": q_boot_gain(y, s, rep, adv, qid)}
        if y2 is not None:
            m2 = (y2 >= 0) & adv
            if m2.sum() > 20 and y2[m2].min() != y2[m2].max():
                rec["adv_y2"] = [round(float(roc_auc_score(y2[m2], s[m2])), 4),
                                 round(float(roc_auc_score(y2[m2], rep[m2])), 4)]
                rec["gain_ci_y2"] = q_boot_gain(y2, s, rep, m2, qid)
        out["slices"][tag] = rec
        print(f"  {tag}: {json.dumps(rec)}", flush=True)
    return out


results = {}
for nm, loader in [("wikiqa", load_wikiqa), ("asnq", load_asnq),
                   ("triviaqa", load_trivia), ("squad", load_squad)]:
    try:
        results[nm] = run(nm, *loader())
    except Exception as e:
        print(f"!! {nm} failed: {type(e).__name__}: {e}", flush=True)
        results[nm] = {"error": f"{type(e).__name__}: {e}"}

with open(os.path.join(HERE, "cross_family_slice_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nwritten cross_family_slice_results.json")
