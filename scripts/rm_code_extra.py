"""Two robustness runs for the reward-model code audit (Section 5.5):

  A. Second RM checkpoint: rescore the SAME MBPP 2x2 candidates with
     OpenAssistant/reward-model-deberta-v3-large-v2 (the C1 screen and its
     committed verdict are properties of (phi, y) and carry over unchanged).
  B. Second code dataset: the frozen recipe applied to HumanEval
     (rm_code_build_humaneval.py), scored with the SAME base RM as the main
     run; the C1 screen for the new setting is committed before scoring.

Both runs report the same objects as the main audit: R2(phi->s), paired
format/construct effects raw vs residualized, cross-format preference
contrasts, and full-population AUC.

Writes ../notes/rm_code_extra.json (+ .committed for the HumanEval screen).
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "notes", "rm_code_extra.json")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)


def format_features(c):
    lines = c.split("\n")
    nonblank = [l for l in lines if l.strip()]
    comments = [l for l in lines if l.strip().startswith("#")]
    ncomm_chars = sum(len(l) for l in comments)
    return [len(c), len(lines), len(c.split()),
            len(comments), ncomm_chars / max(1, len(c)),
            1.0 if '"""' in c else 0.0,
            len(lines) - len(nonblank),
            np.mean([len(l) for l in nonblank]) if nonblank else 0.0,
            max((len(l) for l in nonblank), default=0)]


def score_rm(name, prompt, code, cache_path):
    if os.path.exists(cache_path):
        s = np.load(cache_path)
        assert len(s) == len(code)
        return s
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    print("scoring with", name)
    s = np.zeros(len(code))
    with torch.no_grad():
        for i in range(0, len(code), 8):
            enc = tok(prompt[i:i + 8], code[i:i + 8], return_tensors="pt",
                      truncation=True, max_length=512, padding=True)
            s[i:i + 8] = model(**enc).logits[:, 0].numpy()
            if i % 320 == 0:
                print(f"scored {i}/{len(code)}", flush=True)
    np.save(cache_path, s)
    return s


def analyze(rows, s):
    y = np.array([int(r[2]) for r in rows])
    qid = np.array([int(r[3]) for r in rows])
    cell = np.array([r[4] for r in rows])
    phi = np.array([format_features(r[1]) for r in rows], float)
    s = np.asarray(s, float)
    srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=CV)
    r2 = 1 - float((srep ** 2).sum()) / float(((s - s.mean()) ** 2).sum())
    verbose = np.isin(cell, ["correct-verbose", "buggy-verbose"])
    correct = y == 1

    def paired(u, hi, lo, B=2000):
        per = []
        for t in np.unique(qid):
            a, b = u[(qid == t) & hi], u[(qid == t) & lo]
            if len(a) and len(b):
                per.append(a.mean() - b.mean())
        per = np.array(per)
        bo = [per[rng.integers(0, len(per), len(per))].mean()
              for _ in range(B)]
        return {"mean": round(float(per.mean()), 4),
                "CI": [round(float(np.percentile(bo, 2.5)), 4),
                       round(float(np.percentile(bo, 97.5)), 4)]}

    def effects(u):
        return {"format_correct": paired(u, verbose & correct,
                                         ~verbose & correct),
                "format_buggy": paired(u, verbose & ~correct,
                                       ~verbose & ~correct),
                "construct_terse": paired(u, correct & ~verbose,
                                          ~correct & ~verbose),
                "construct_verbose": paired(u, correct & verbose,
                                            ~correct & verbose)}

    def cross_pref(u, hi_cell, lo_cell, B=2000):
        per = []
        for t in np.unique(qid):
            a = u[(qid == t) & (cell == hi_cell)]
            b = u[(qid == t) & (cell == lo_cell)]
            if len(a) and len(b):
                per.append(float((a[:, None] > b[None, :]).mean()))
        per = np.array(per)
        bo = [per[rng.integers(0, len(per), len(per))].mean()
              for _ in range(B)]
        return [round(float(per.mean()), 4),
                [round(float(np.percentile(bo, 2.5)), 4),
                 round(float(np.percentile(bo, 97.5)), 4)]]

    return {"n": int(len(y)), "n_problems": int(len(np.unique(qid))),
            "C2_R2_phi_to_s": round(r2, 4),
            "effects_raw": effects(s), "effects_residualized": effects(srep),
            "A_full": [round(float(roc_auc_score(y, s)), 4),
                       round(float(roc_auc_score(y, srep)), 4)],
            "cross_pref_ct_vs_bv": {"raw": cross_pref(s, "correct-terse",
                                                      "buggy-verbose"),
                                    "rep": cross_pref(srep, "correct-terse",
                                                      "buggy-verbose")},
            "cross_pref_cv_vs_bt": {"raw": cross_pref(s, "correct-verbose",
                                                      "buggy-terse"),
                                    "rep": cross_pref(srep, "correct-verbose",
                                                      "buggy-terse")}}


res = {}

# ---- A. second RM checkpoint on the SAME MBPP candidates ----
rows = json.load(open(os.path.join(CACHE, "mbpp_rows.json")))
s_large = score_rm("OpenAssistant/reward-model-deberta-v3-large-v2",
                   [r[0] for r in rows], [r[1] for r in rows],
                   os.path.join(CACHE, "mbpp_rm_large_s.npy"))
res["mbpp_large_v2"] = analyze(rows, s_large)

# ---- B. HumanEval with the base RM; C1 committed before scoring ----
he = json.load(open(os.path.join(CACHE, "humaneval_rows.json")))
y = np.array([int(r[2]) for r in he])
phi = np.array([format_features(r[1]) for r in he], float)
art = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
prob = cross_val_predict(art, phi, y, cv=CV, method="predict_proba")[:, 1]
c1 = float(roc_auc_score(y, prob))
committed = {"C1_screen": round(c1, 4),
             "committed_prediction": ("improvement expected" if c1 >= 0.65
                                      else "C1 fails: screen refuses "
                                      "slice-level repair; paired-"
                                      "intervention analysis as in MBPP")}
print("HUMANEVAL COMMITTED BEFORE SCORING:", json.dumps(committed))
json.dump(committed, open(OUT + ".committed", "w"), indent=1)
s_he = score_rm("OpenAssistant/reward-model-deberta-v3-base",
                [r[0] for r in he], [r[1] for r in he],
                os.path.join(CACHE, "humaneval_rm_s.npy"))
res["humaneval_base"] = analyze(he, s_he)
res["humaneval_base"]["C1"] = round(c1, 4)
res["humaneval_base"]["committed"] = committed

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
