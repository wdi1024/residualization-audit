"""Internal-review round 4 (E items): screening-statistic CIs, ridge-alpha
sweep, and rank-Gaussian decorrelation.

Per QA setting (question-grouped protocol, identical row construction to
grouped_full.py) and per NLI setting (row-level, as in the paper):
  1. Bootstrap CIs (query-clustered for QA, example-level otherwise) for the
     three screening statistics C1, R2(phi->s), Delta_adv. Predictions are
     held fixed; the evaluation sample is resampled (evaluation-resampling
     CIs, the same convention as every other CI in the paper).
  2. Ridge alpha sweep over {0.1, 1, 10, 100}: adversarial-slice gain and
     R_lin(s_rep) at each alpha (full refit per alpha).
  3. Rank-Gaussian score representation: R_lin before/after repair and the
     slice gain under s_rg = Phi^{-1}(rank/(n+1)) (the paper's recommended
     robust representation reports gains in Table 10; this adds R_lin).
Also: toxicity Delta_adv / R2 CIs (row-level), since its gate margin is at
the third decimal.
"""
import json
import os
import re

import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack, csr_matrix
from scipy.stats import norm, rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
OUT_PATH = os.path.join(HERE, "review_r4_results.json")
rng = np.random.default_rng(0)
NBOOT = 1000
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


# ---------- loaders (identical to grouped_full.py) ----------

def load_wikiqa():
    d = load_dataset("microsoft/wiki_qa", split="train")
    rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"), np.array([uniq[qq] for qq in q])


def load_asnq():
    rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
    q = [a for a, _, _ in rows]
    ans = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/asnq_ce_s_8000.npy"), np.array([uniq[qq] for qq in q])


def load_trivia():
    rows = np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)
    q = [r[0] for r in rows]
    ans = [r[1] for r in rows]
    y = np.array([int(r[2]) for r in rows])
    uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
    return q, ans, y, np.load(f"{CACHE}/triviaqa_ce_s.npy"), np.array([uniq[qq] for qq in q])


def load_squad():
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
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows]), np.load(f"{CACHE}/squad_ce_s_8000.npy"),
            np.array(qid[:8000]))


def load_duorc():
    d = load_dataset("ibm/duorc", "SelfRC", split="validation")
    rows, qid, seen_q = [], [], 0
    for r in d:
        sents = sent_split(r["plot"])
        if not (3 <= len(sents) <= 25):
            continue
        if r["no_answer"] or not r["answers"]:
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
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows]), np.load(f"{CACHE}/duorc_ce_s_8000.npy"),
            np.array(qid[:8000]))


# ---------- bootstrap helpers (predictions fixed, sample resampled) ----------

def boot_stats(y, s, art, pred, adv, qid, nboot=NBOOT):
    """CIs for C1 (AUC(y, art)), R2(phi->s) (1 - SSres/SStot with fixed
    cross-fitted pred), Delta_adv (AUC_full - AUC_adv, fixed slice)."""
    if qid is None:
        qid = np.arange(len(y))
    qs = np.unique(qid)
    by_q = {qq: np.where(qid == qq)[0] for qq in qs}
    c1s, r2s, dds = [], [], []
    for _ in range(nboot):
        pick = rng.choice(qs, size=len(qs), replace=True)
        idx = np.concatenate([by_q[qq] for qq in pick])
        yy, ss, aa, pp, vv = y[idx], s[idx], art[idx], pred[idx], adv[idx]
        if yy.min() == yy.max():
            continue
        c1s.append(roc_auc_score(yy, aa))
        r2s.append(1 - float(((ss - pp) ** 2).sum())
                   / max(1e-12, float(((ss - ss.mean()) ** 2).sum())))
        if yy[vv].min() != yy[vv].max():
            dds.append(roc_auc_score(yy, ss) - roc_auc_score(yy[vv], ss[vv]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)]
    return {"C1_CI": ci(c1s), "R2_CI": ci(r2s), "delta_adv_CI": ci(dds)}


def rank_gauss(s):
    return norm.ppf(rankdata(s) / (len(s) + 1))


def run_setting(name, phi, dense, y, s, qid, scaled_probe=False):
    """dense: features for the artifact model; phi: features for the repair."""
    grouped = qid is not None
    cv = GroupKFold(5) if grouped else KFold(5, shuffle=True, random_state=0)
    kw = dict(cv=cv, groups=qid) if grouped else dict(cv=cv)

    if grouped:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif scaled_probe:  # toxicity's original recipe (c2_metrics_uniform.py)
        pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    else:
        pipe = LogisticRegression(max_iter=1000)  # paper's row-level NLI recipe: no scaler
    art = cross_val_predict(pipe, dense, y, method="predict_proba", **kw)[:, 1]
    chat = cross_val_predict(pipe, dense, y, method="decision_function", **kw)
    c1 = roc_auc_score(y, art)
    if grouped:
        thr = np.quantile(art, 1 - y.mean())
        z = art > thr
    else:
        z = art > 0.5
    adv = z != (y == 1)

    pred1 = cross_val_predict(Ridge(alpha=1.0), phi, s, **kw)
    r2 = 1 - float(((s - pred1) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
    dadv = float(roc_auc_score(y, s) - roc_auc_score(y[adv], s[adv]))

    res = {"n": len(y), "n_adv": int(adv.sum()),
           "C1": round(float(c1), 4), "R2": round(r2, 4),
           "delta_adv": round(dadv, 4)}
    res.update(boot_stats(y, s, art, pred1, adv, qid))

    A = lambda m, u: float(roc_auc_score(y[m], u[m]))
    res["alpha_sweep"] = {}
    for a in ALPHAS:
        pr = pred1 if a == 1.0 else cross_val_predict(Ridge(alpha=a), phi, s, **kw)
        srep = s - pr
        res["alpha_sweep"][str(a)] = {
            "gain_adv": round(A(adv, srep) - A(adv, s), 4),
            "A_adv": [round(A(adv, s), 4), round(A(adv, srep), 4)],
            "Rlin_rep": round(abs(float(np.corrcoef(srep, chat)[0, 1])), 4),
            "R2": round(1 - float(((s - pr) ** 2).sum())
                        / float(((s - s.mean()) ** 2).sum()), 4)}

    s_rg = rank_gauss(s)
    pr_rg = cross_val_predict(Ridge(alpha=1.0), phi, s_rg, **kw)
    srep_rg = s_rg - pr_rg
    res["rank_gauss"] = {
        "Rlin_raw": round(abs(float(np.corrcoef(s_rg, chat)[0, 1])), 4),
        "Rlin_rep": round(abs(float(np.corrcoef(srep_rg, chat)[0, 1])), 4),
        "gain_adv": round(A(adv, srep_rg) - A(adv, s_rg), 4)}

    print(name, json.dumps(res, indent=1), flush=True)
    return res


def qa_features(q, ans):
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    return hstack([Xa, csr_matrix(dense)]).tocsr(), dense


def run_nli(name, hf_id, split, cache_file, label_map):
    d = load_dataset(hf_id, split=split)
    key_p, key_h, key_l = label_map
    rows = [(r[key_p], r[key_h], 1 if r[key_l] == 0 else 0)
            for r in d if r[key_l] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load(f"{CACHE}/{cache_file}")
    assert len(s) == len(y), (len(s), len(y))
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return run_setting(name, Xh, Xh, y, s, None)


def run_sick():
    d = load_dataset("yangwang825/sick", split="test")
    rows = [(r["text1"], r["text2"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:2400]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    s = np.load(f"{CACHE}/sick_s.npy")
    assert len(s) == len(y), (len(s), len(y))
    print("sick sanity A_full_raw:", round(float(roc_auc_score(y, s)), 3), "(paper: 0.967)")
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return run_setting("sick", Xh, Xh, y, s, None)


IDENT = ["black", "white", "muslim", "jewish", "christian", "gay", "lesbian",
         "homosexual", "trans", "transgender", "women", "woman", "female",
         "men", "man", "male", "immigrant", "mexican", "islam", "muslims",
         "jews", "gays", "queer", "asian", "african", "catholic", "atheist",
         "feminist"]


def run_tox():
    N = 4000
    d = load_dataset("google/civil_comments", split="train", streaming=True)
    pos, neg = [], []
    for r in d:
        t, tox = r["text"], r["toxicity"]
        if tox is None or not t.strip():
            continue
        lab = 1 if tox >= 0.5 else (0 if tox <= 0.1 else None)
        if lab is None:
            continue
        (pos if lab == 1 else neg).append(t)
        if len(pos) >= N // 2 and len(neg) >= N // 2:
            break
    texts = pos[:N // 2] + neg[:N // 2]
    y = np.array([1] * (N // 2) + [0] * (N // 2))
    s = np.load(f"{CACHE}/tox_s_4000.npy")

    def ident_counts(t):
        tl = t.lower()
        return [len(re.findall(r"\b" + re.escape(w) + r"\b", tl)) for w in IDENT]
    ic = np.array([ident_counts(t) for t in texts], float)
    ln = np.array([len(t.split()) for t in texts], float)[:, None]
    phi = csr_matrix(np.hstack([ic, ln, ic.sum(1, keepdims=True)]))
    return run_setting("toxicity", phi, phi, y, s, None, scaled_probe=True)


def main():
    out = {}
    for name, loader in [("wikiqa", load_wikiqa), ("asnq", load_asnq),
                         ("triviaqa", load_trivia), ("squad", load_squad),
                         ("duorc", load_duorc)]:
        q, ans, y, s, qid = loader()
        phi, dense = qa_features(q, ans)
        out[name] = run_setting(name, phi, dense, y, s, qid)
        json.dump(out, open(OUT_PATH, "w"), indent=1)

    out["snli"] = run_nli("snli", "stanfordnlp/snli", "validation",
                          "snli_s_2400.npy", ("premise", "hypothesis", "label"))
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    out["sick"] = run_sick()
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    out["toxicity"] = run_tox()
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("done ->", OUT_PATH)


if __name__ == "__main__":
    main()
