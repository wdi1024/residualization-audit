"""Second disjoint-reading setting (internal review round 11, item 2):
answer-sentence selection built from Natural Questions validation, whose
5-way annotations supply a genuine disjoint second reading -- the same
separation as the SQuAD dual evaluation, on an independently annotated
dataset.

Construction mirrors the SQuAD builder: for each question, the context is
the first annotator's long-answer paragraph (HTML tokens stripped),
sentence-split; y1 labels the sentence containing annotator 1's short
answer, y2 labels sentences containing any DISJOINT annotator's (2-5)
short answer. Questions kept only if annotator 1 has a short answer, at
least one disjoint annotator has one, the context has 3-25 sentences, and
y1 marks exactly one sentence. Scoring: the MS MARCO MiniLM-L6
cross-encoder, as in every QA setting. Pipeline, phi, slice recipe, and
question-grouped protocol identical to grouped_full.py.

Everything downstream of construction is the frozen recipe; this setting
was built after all freezes and is reported as a post-hoc replication.
Scores cached to ../cache/nq_ce_s.npy; rows to ../cache/nq_rows.json.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
ROWS_CACHE = os.path.join(CACHE, "nq_rows.json")
SCORE_CACHE = os.path.join(CACHE, "nq_ce_s.npy")
OUT = os.path.join(HERE, "nq_dual_reading.json")
TARGET_ROWS = 8000


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


# ---------------- build rows ----------------
if os.path.exists(ROWS_CACHE):
    rows = json.load(open(ROWS_CACHE))
    print(f"loaded {len(rows)} cached rows", flush=True)
else:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/natural_questions",
                      "default", split="validation", streaming=True)
    rows = []  # [question, sentence, y1, qid, y2]
    seen_q = 0
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned % 500 == 0:
            print(f"scanned {scanned}, kept {seen_q} questions, "
                  f"{len(rows)} rows", flush=True)
        anns = ex["annotations"]
        # annotator 1 = first annotation; disjoint = the rest
        sa = anns["short_answers"]
        if not sa or not sa[0]["text"]:
            continue
        a1_texts = [t for t in sa[0]["text"] if t.strip()]
        if not a1_texts:
            continue
        rest_texts = []
        for k in range(1, len(sa)):
            rest_texts += [t for t in sa[k]["text"] if t.strip()]
        if not rest_texts:
            continue
        la = anns["long_answer"][0]
        if la["start_token"] < 0:
            continue
        toks = ex["document"]["tokens"]
        words = [w for w, h in zip(
            toks["token"][la["start_token"]:la["end_token"]],
            toks["is_html"][la["start_token"]:la["end_token"]]) if not h]
        context = " ".join(words)
        sents = sent_split(context)
        if not (3 <= len(sents) <= 25):
            continue
        pos1 = [i for i, s_ in enumerate(sents)
                if any(a in s_ for a in a1_texts)]
        if len(pos1) != 1:
            continue
        q = ex["question"]["text"]
        for i, s_ in enumerate(sents):
            y2 = 1 if any(a in s_ for a in rest_texts) else 0
            rows.append([q, s_, 1 if i == pos1[0] else 0, seen_q, y2])
        seen_q += 1
        if len(rows) >= TARGET_ROWS:
            break
    rows = rows[:TARGET_ROWS]
    json.dump(rows, open(ROWS_CACHE, "w"))
    print(f"built {len(rows)} rows from {seen_q} questions "
          f"({scanned} scanned)", flush=True)

q = [r[0] for r in rows]
ans = [r[1] for r in rows]
y = np.array([r[2] for r in rows])
qid = np.array([r[3] for r in rows])
y2 = np.array([r[4] for r in rows])
print(f"n={len(y)} questions={len(np.unique(qid))} pos_rate={y.mean():.3f} "
      f"y1/y2 agreement={float((y == y2).mean()):.4f}", flush=True)

# ---------------- score with the MS MARCO cross-encoder ----------------
if os.path.exists(SCORE_CACHE):
    s = np.load(SCORE_CACHE)
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    torch.set_num_threads(6)
    s = np.zeros(len(y))
    B = 64
    with torch.no_grad():
        for b in range(0, len(y), B):
            enc = tok(q[b:b+B], ans[b:b+B], truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            s[b:b+B] = model(**enc).logits[:, 0].numpy()
            if b % 1280 == 0:
                print(f"scored {b}/{len(y)}", flush=True)
    np.save(SCORE_CACHE, s)
print("scores ready", flush=True)

# ---------------- frozen pipeline, grouped ----------------
sys.path.insert(0, HERE)
from review_r4_experiments import run_setting, qa_features  # noqa: E402
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

phi, dense = qa_features(q, ans)
res = run_setting("nq_dual", phi, dense, y, s, qid)

# disjoint-reading evaluation: slice and pipeline from y1, evaluation on y2
cv = GroupKFold(5)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
art = cross_val_predict(pipe, dense, y, cv=cv, groups=qid,
                        method="predict_proba")[:, 1]
thr = np.quantile(art, 1 - y.mean())
adv = (art > thr) != (y == 1)
srep = s - cross_val_predict(Ridge(alpha=1.0), phi, s, cv=cv, groups=qid)

rng = np.random.default_rng(0)


def gain_ci(yy_all):
    qs = np.unique(qid[adv])
    by_q = {qq: np.where(adv & (qid == qq))[0] for qq in qs}
    gains = []
    for _ in range(1000):
        pick = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([by_q[qq] for qq in pick])
        yy = yy_all[idx]
        if yy.min() == yy.max():
            continue
        gains.append(roc_auc_score(yy, srep[idx]) - roc_auc_score(yy, s[idx]))
    return [round(float(np.percentile(gains, 2.5)), 4),
            round(float(np.percentile(gains, 97.5)), 4)]


res["agreement_y1_y2"] = round(float((y == y2).mean()), 4)
res["gain_same_label"] = round(float(
    roc_auc_score(y[adv], srep[adv]) - roc_auc_score(y[adv], s[adv])), 4)
res["gain_same_label_ci"] = gain_ci(y)
mask_ok = y2[adv]
if len(np.unique(y2[adv])) == 2:
    res["adv_y2"] = [round(float(roc_auc_score(y2[adv], s[adv])), 4),
                     round(float(roc_auc_score(y2[adv], srep[adv])), 4)]
    res["gain_disjoint"] = round(res["adv_y2"][1] - res["adv_y2"][0], 4)
    res["gain_disjoint_ci"] = gain_ci(y2)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items()
                  if not isinstance(v, dict)}, indent=1))
print("wrote", OUT)
