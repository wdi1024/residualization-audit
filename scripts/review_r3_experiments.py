"""Round-3 review experiments.

E6  linear-INDEX certificate (decision_function, exactly w^T phi after
    standardization) for SNLI and SICK, completing the set so the paper's
    primary certificate can be the linear index everywhere.
E7  OBSERVABLE-subgroup analysis: restrict to the phi-only flagged subgroup
    z(x)=1[a_hat(x) > thr] (computable without y at deployment) and compare
    A(s) vs A(s_rep) there, for the QA settings and SNLI/SICK.
    This is the deployment-relevant, non-circular counterpart of the
    label-defined adversarial slice.

Writes ../notes/review_r3_experiments.json.
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
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def qa_build(rows):
    q = [r[0] for r in rows]
    ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    return y, dense, phi


def load_snli():
    from datasets import load_dataset
    d = load_dataset("stanfordnlp/snli", split="validation")
    rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load(f"{CACHE}/snli_s_2400.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return y, Xh, Xh, s, 0.5


def load_sick():
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            "yangwang825%2Fsick&config=default&split=test")
    cachef = "/tmp/sick_rows.json"
    if os.path.exists(cachef):
        rows = json.load(open(cachef))
    else:
        rows = []
        for off in range(0, 4906, 100):
            out = subprocess.run(["curl", "-sL", "--max-time", "60",
                                  f"{base}&offset={off}&length=100"],
                                 capture_output=True, check=True).stdout
            rows += [x["row"] for x in json.loads(out)["rows"]]
        json.dump(rows, open(cachef, "w"))
    pairs = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
             for r in rows if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in pairs]
    y = np.array([c for _, _, c in pairs])
    s = np.load(f"{CACHE}/sick_s.npy")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return y, Xh, Xh, s, 0.5


def analyze(name, y, artX, phi, s, thr_rule, standardize_art):
    s = np.asarray(s, float)
    if standardize_art:
        art_model = make_pipeline(StandardScaler(with_mean=False),
                                  LogisticRegression(max_iter=1000))
    else:
        art_model = LogisticRegression(max_iter=1000)
    prob = cross_val_predict(art_model, artX, y, cv=CV,
                             method="predict_proba")[:, 1]
    index = cross_val_predict(art_model, artX, y, cv=CV,
                              method="decision_function")
    srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    thr = (np.quantile(prob, 1 - y.mean()) if thr_rule == "rate" else 0.5)
    flagged = prob > thr                       # phi-only, observable
    res = {"n": len(y), "n_flagged": int(flagged.sum()),
           "cert_index_raw": abs(float(np.corrcoef(s, index)[0, 1])),
           "cert_index_rep": abs(float(np.corrcoef(srep, index)[0, 1])),
           "cert_prob_raw": abs(float(np.corrcoef(s, prob)[0, 1])),
           "cert_prob_rep": abs(float(np.corrcoef(srep, prob)[0, 1]))}
    for tag, m in [("flagged", flagged), ("unflagged", ~flagged)]:
        if y[m].min() != y[m].max():
            res[f"A_raw_{tag}"] = float(roc_auc_score(y[m], s[m]))
            res[f"A_rep_{tag}"] = float(roc_auc_score(y[m], srep[m]))
    res["A_raw_full"] = float(roc_auc_score(y, s))
    res["A_rep_full"] = float(roc_auc_score(y, srep))
    print(name, json.dumps(res, indent=1))
    return res


out = {}
# --- NLI settings (E6 focus; E7 also reported)
y, artX, phi, s, thr = load_snli()
out["snli"] = analyze("snli", y, artX, phi, s, thr, standardize_art=False)
y, artX, phi, s, thr = load_sick()
out["sick"] = analyze("sick", y, artX, phi, s, thr, standardize_art=False)

# --- QA settings (E7 focus; index certificate re-confirmed with dense art)
from datasets import load_dataset
d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
y, dense, phi = qa_build(rows)
s = np.load(f"{CACHE}/wikiqa_ce_s_8000.npy")
out["wikiqa"] = analyze("wikiqa", y, dense, phi, s, "rate", standardize_art=True)

rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
y, dense, phi = qa_build(rows)
s = np.load(f"{CACHE}/asnq_ce_s_8000.npy")
out["asnq"] = analyze("asnq", y, dense, phi, s, "rate", standardize_art=True)

rows = [list(r) for r in np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)]
y, dense, phi = qa_build(rows)
s = np.load(f"{CACHE}/triviaqa_ce_s.npy")
out["triviaqa"] = analyze("triviaqa", y, dense, phi, s, "rate",
                          standardize_art=True)

json.dump(out, open(os.path.join(HERE, "..", "notes",
                                 "review_r3_experiments.json"), "w"), indent=1)
print("saved notes/review_r3_experiments.json")
