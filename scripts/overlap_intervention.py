"""Rule-based overlap intervention (the reviews' standing demand): manipulate
question--candidate lexical overlap while holding relevance fixed, and test
whether the raw score follows the manipulated overlap while the residualized
score does not.

Design (SQuAD sentence selection, same build as the held-out run):
  A. overlap-DOWN on positives: replace question-overlapping words in the
     answer-bearing sentence with WordNet synonyms (never touching the answer
     span tokens), so the sentence still contains the answer (y=1 preserved
     by construction) but shares fewer words with the question.
  B. overlap-UP on negatives: append a short clause built from question
     content words to a non-answer sentence (y=0 preserved: the answer string
     is still absent), raising overlap without adding the answer.

Both interventions are deterministic, rule-based, and content-checked
(answer containment re-verified after editing). The ridge g(phi) is trained
once on the ORIGINAL 8k rows (the frozen operator) and applied to original
and edited candidates alike; the CE scores the edited text.

Prediction under the paper's reading: the raw score shifts with the overlap
manipulation (it rides the channel); the residualized score shifts less
(the channel component is subtracted); relevance is fixed by construction.

Writes ../notes/overlap_intervention.json.
"""
import json
import os
import re

import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from scipy.sparse import hstack, csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
CV = KFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)

STOP = set("the a an of to in and or is are was were be been on for with as at "
           "by from that this it its他 he she they what which who when where "
           "how why did does do".split())


def overlap(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def synonym(word):
    for syn in wn.synsets(word.lower()):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower() and " " not in w and w.isalpha():
                return w
    return None


# ---------------- rebuild the held-out candidates ----------------
from datasets import load_dataset
d = load_dataset("squad", split="validation")
rows, qid = [], []
seen_q = 0
for r in d:
    sents = sent_split(r["context"])
    if not (3 <= len(sents) <= 25):
        continue
    ans = r["answers"]["text"][0]
    pos = [i for i, s in enumerate(sents) if ans in s]
    if len(pos) != 1:
        continue
    for i, s in enumerate(sents):
        rows.append((r["question"], s, 1 if i == pos[0] else 0, ans))
        qid.append(seen_q)
    seen_q += 1
    if len(rows) >= 8000:
        break
rows = rows[:8000]
q = [r[0] for r in rows]
ans_s = [r[1] for r in rows]
y = np.array([r[2] for r in rows])
answers = [r[3] for r in rows]

# frozen operator: ridge on ORIGINAL phi -> s
def build_phi(qs, cs):
    ov = np.array([overlap(a, b) for a, b in zip(qs, cs)])[:, None]
    la = np.array([len(x.split()) for x in cs], float)[:, None]
    lq = np.array([len(x.split()) for x in qs], float)[:, None]
    return ov, la, lq

s_orig = np.load(os.path.join(CACHE, "squad_ce_s_8000.npy"))
ovO, laO, lqO = build_phi(q, ans_s)
denseO = np.hstack([ovO, laO, lqO, laO - lqO])
vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit(ans_s)
XaO = vec.transform(ans_s)
phiO = hstack([XaO, csr_matrix(denseO)]).tocsr()
g_model = Ridge(alpha=1.0).fit(phiO, s_orig)   # frozen operator fit once

# ---------------- interventions ----------------
def edit_down(question, sent, answer):
    qwords = {w.lower() for w in question.split()} - STOP
    out = []
    changed = 0
    for w in sent.split():
        core = re.sub(r"\W", "", w)
        if (core.lower() in qwords and core.lower() not in answer.lower()
                and answer not in w):
            syn = synonym(core)
            if syn:
                out.append(w.replace(core, syn))
                changed += 1
                continue
        out.append(w)
    return " ".join(out), changed


def edit_up(question, sent):
    kws = [w for w in question.split()
           if w.lower() not in STOP and re.sub(r"\W", "", w).isalpha()][:3]
    if not kws:
        return None
    return sent.rstrip(".") + ", regarding " + " ".join(kws) + "."


down_idx = [i for i in range(len(rows)) if y[i] == 1][:600]
up_pool = [i for i in range(len(rows)) if y[i] == 0]
up_idx = list(rng.choice(up_pool, 600, replace=False))

edited, kinds, orig_ids = [], [], []
for i in down_idx:
    e, ch = edit_down(q[i], ans_s[i], answers[i])
    if ch >= 1 and answers[i] in e:            # answer preserved => y=1 intact
        edited.append(e); kinds.append("down"); orig_ids.append(i)
for i in up_idx:
    e = edit_up(q[i], ans_s[i])
    if e and answers[i] not in e:              # answer absent => y=0 intact
        edited.append(e); kinds.append("up"); orig_ids.append(i)
kinds = np.array(kinds); orig_ids = np.array(orig_ids)
print(f"edited: down={int((kinds=='down').sum())} up={int((kinds=='up').sum())}")

# score edited candidates
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
s_new = np.asarray(ce.predict([(q[i], e) for i, e in zip(orig_ids, edited)],
                              batch_size=64), float)

# phi for edited candidates, frozen vectorizer + frozen ridge
qe = [q[i] for i in orig_ids]
ovE, laE, lqE = build_phi(qe, edited)
denseE = np.hstack([ovE, laE, lqE, laE - lqE])
XaE = vec.transform(edited)
phiE = hstack([XaE, csr_matrix(denseE)]).tocsr()

srep_orig = s_orig - g_model.predict(phiO)
srep_new = s_new - g_model.predict(phiE)

res = {}
for kind in ("down", "up"):
    m = kinds == kind
    ids = orig_ids[m]
    d_ov = float(np.mean(ovE[m, 0] - ovO[ids, 0]))
    d_raw = float(np.mean(s_new[m] - s_orig[ids]))
    d_rep = float(np.mean(srep_new[m] - srep_orig[ids]))
    sd_raw = float(np.std(s_new[m] - s_orig[ids]) / np.sqrt(m.sum()))
    sd_rep = float(np.std(srep_new[m] - srep_orig[ids]) / np.sqrt(m.sum()))
    res[kind] = {"n": int(m.sum()), "delta_overlap": d_ov,
                 "delta_raw_score": d_raw, "se_raw": sd_raw,
                 "delta_residualized": d_rep, "se_rep": sd_rep,
                 "attenuation": float(1 - abs(d_rep) / max(1e-9, abs(d_raw)))}
res["note"] = ("relevance fixed by construction: down-edits keep the answer "
               "span verbatim (y=1), up-edits never introduce it (y=0)")
json.dump(res, open(os.path.join(HERE, "..", "notes",
                                 "overlap_intervention.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
