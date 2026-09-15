"""Complete question-grouped rerun of every main-text QA quantity.

Extends qa_grouped_all/squad_duorc_grouped (same row construction, same recipe)
to the full set the paper quotes: Delta_adv, before/after alignments on the three
populations, query-clustered bootstrap CIs, certificate R_lin(+CI), SQuAD dual
reading / high-overlap quartile / feature split, TriviaQA decision floor,
WikiQA scorer replications (cached scores), and query-level ranking metrics
(slice MRR / per-query AUC) requested by mock review #11.
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
            np.array(qid[:8000]), None)

def q_boot_gain(y, raw, rep, mask, qid, nboot=NBOOT):
    """Query-clustered bootstrap CI for AUC(y,rep)-AUC(y,raw) on mask rows."""
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

def q_boot_abscorr(a, b, qid, nboot=NBOOT):
    qs = np.unique(qid)
    by_q = {qq: np.where(qid == qq)[0] for qq in qs}
    vals = []
    for _ in range(nboot):
        pick = rng.choice(qs, size=len(qs), replace=True)
        idx = np.concatenate([by_q[qq] for qq in pick])
        vals.append(abs(float(np.corrcoef(a[idx], b[idx])[0, 1])))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

def rankings(y, s_raw, s_rep, qid, adv):
    """Query-level ranking: MRR + mean per-query AUC, all queries vs slice-touching."""
    out = {}
    for scope, qsel in [("all", None), ("slice", set(np.unique(qid[adv])))]:
        mrr_r, mrr_p, pq_r, pq_p = [], [], [], []
        for qq in np.unique(qid):
            if qsel is not None and qq not in qsel:
                continue
            idx = np.where(qid == qq)[0]
            yy = y[idx]
            if yy.max() == 0 or len(idx) < 2:
                continue
            for scores, mrr_acc, pq_acc in [(s_raw, mrr_r, pq_r), (s_rep, mrr_p, pq_p)]:
                order = np.argsort(-scores[idx])
                rank = 1 + int(np.where(yy[order] == 1)[0][0])
                mrr_acc.append(1.0 / rank)
                if yy.min() == 0:
                    pq_acc.append(roc_auc_score(yy, scores[idx]))
        out[scope] = {"n_queries": len(mrr_r),
                      "MRR": [round(float(np.mean(mrr_r)), 4), round(float(np.mean(mrr_p)), 4)],
                      "perq_AUC": [round(float(np.mean(pq_r)), 4), round(float(np.mean(pq_p)), 4)]}
    return out

def run(name, q, ans, y, s, qid, y2, extra_scores=None):
    print(f"\n=== {name}: n={len(y)} pos={y.mean():.3f} queries={len(np.unique(qid))}", flush=True)
    ov = np.array([overlap(a, b) for a, b in zip(q, ans)])[:, None]
    la = np.array([len(x.split()) for x in ans], float)[:, None]
    lq = np.array([len(x.split()) for x in q], float)[:, None]
    dense = np.hstack([ov, la, lq, la - lq])
    Xa = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(ans)
    phi = hstack([Xa, csr_matrix(dense)]).tocsr()
    cv = GroupKFold(5)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    art = cross_val_predict(pipe, dense, y, cv=cv, groups=qid, method="predict_proba")[:, 1]
    chat = cross_val_predict(pipe, dense, y, cv=cv, groups=qid, method="decision_function")
    c1 = roc_auc_score(y, art)
    thr = np.quantile(art, 1 - y.mean())
    z = art > thr
    adv = z != (y == 1)

    pred = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, groups=qid)
    s_rep = s - pred
    r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())

    A = lambda m, u: round(float(roc_auc_score(y[m], u[m])), 4)
    full = np.ones(len(y), bool)
    res = {
        "C1": round(c1, 4), "R2": round(r2, 4),
        "delta_adv": round(float(roc_auc_score(y, s) - roc_auc_score(y[adv], s[adv])), 4),
        "n_adv": int(adv.sum()),
        "adv": [A(adv, s), A(adv, s_rep)], "adv_gain_ci": q_boot_gain(y, s, s_rep, adv, qid),
        "full": [A(full, s), A(full, s_rep)],
        "flag": [A(z, s), A(z, s_rep)] if len(set(y[z])) > 1 else None,
        "cert": {"raw": round(abs(float(np.corrcoef(s, chat)[0, 1])), 4),
                 "rep": round(abs(float(np.corrcoef(s_rep, chat)[0, 1])), 4),
                 "rep_ci": q_boot_abscorr(s_rep, chat, qid)},
        "floor": [round(max(roc_auc_score(z, s), 1 - roc_auc_score(z, s)), 4),
                  round(max(roc_auc_score(z, s_rep), 1 - roc_auc_score(z, s_rep)), 4)],
        "rankings": rankings(y, s, s_rep, qid, adv),
    }

    if name == "squad":
        m2 = (y2 >= 0) & adv
        res["dual_reading"] = {
            "agreement": round(float((y2[y2 >= 0] == y[y2 >= 0]).mean()), 4),
            "adv_y2": [A2 := round(float(roc_auc_score(y2[m2], s[m2])), 4),
                       round(float(roc_auc_score(y2[m2], s_rep[m2])), 4)],
            "gain_ci": q_boot_gain(y2, s, s_rep, m2, qid),
        }
        qmask = ov[:, 0] >= np.quantile(ov[:, 0], 0.75)
        res["hi_overlap_quartile"] = [A(qmask, s), A(qmask, s_rep)]
        splits = {}
        for tag, sub in [("overlap_only", csr_matrix(ov)),
                         ("answer_only", hstack([Xa, csr_matrix(la)]).tocsr())]:
            p2 = cross_val_predict(Ridge(alpha=1.0), sub, s, cv=cv, groups=qid)
            r_2 = s - p2
            splits[tag] = round(float(roc_auc_score(y[adv], r_2[adv]) - roc_auc_score(y[adv], s[adv])), 4)
        res["feature_split_adv_gain"] = splits

    if extra_scores:
        res["scorers"] = {}
        for tag, s2 in extra_scores.items():
            p2 = cross_val_predict(Ridge(alpha=1.0), phi, s2, cv=cv, groups=qid)
            rp = s2 - p2
            r2b = 1 - float(((s2 - p2) ** 2).sum()) / float(((s2 - s2.mean()) ** 2).sum())
            res["scorers"][tag] = {
                "R2": round(r2b, 4),
                "adv": [round(float(roc_auc_score(y[adv], s2[adv])), 4),
                        round(float(roc_auc_score(y[adv], rp[adv])), 4)],
                "gain_ci": q_boot_gain(y, s2, rp, adv, qid),
            }
    print(json.dumps(res, indent=1)[:1200], flush=True)
    return res

results = {}
q, ans, y, s, qid, _ = load_wikiqa()
results["wikiqa"] = run("wikiqa", q, ans, y, s, qid, None, extra_scores={
    "qnli": np.load(f"{CACHE}/wikiqa_qnli_s_8000.npy"),
    "stsb": np.load(f"{CACHE}/wikiqa_stsb_s_8000.npy")})
results["asnq"] = run("asnq", *load_asnq())
results["triviaqa"] = run("triviaqa", *load_trivia())
results["squad"] = run("squad", *load_squad())
results["duorc"] = run("duorc", *load_duorc())
json.dump(results, open(os.path.join(HERE, "grouped_full_results.json"), "w"), indent=1)
print("\nwrote grouped_full_results.json", flush=True)
