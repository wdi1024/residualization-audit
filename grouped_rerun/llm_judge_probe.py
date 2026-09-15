"""Modern-scorer probe: an LLM judge on the WikiQA setting (internal review
round 8, item 3 -- 'does a current-generation judge carry a linear format
channel?').

Design: identical rows, phi, slice recipe, and question-grouped protocol as
the WikiQA setting (grouped_full.py); only the scorer changes. The judge is
Claude Haiku (claude-haiku-4-5), scoring each candidate sentence's relevance
to the question on a 0-100 scale (temperature 0, single-number output,
score = the number). The screen then runs exactly as everywhere else:
C1 (a property of the data, unchanged), R2(phi->s), Delta_adv, and -- if C2
passes -- cross-fitted ridge repair with slice gain and R_lin.

Scores are cached to ../cache/wikiqa_haiku_s_8000.npy (resumable: NaN rows
are re-scored on rerun). Failures after retries are left NaN and excluded.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
SCORE_CACHE = os.path.join(CACHE, "wikiqa_haiku_s_8000.npy")
OUT = os.path.join(HERE, "llm_judge_probe.json")
MODEL = "claude-haiku-4-5-20251001"
N_WORKERS = 12

from datasets import load_dataset

d = load_dataset("microsoft/wiki_qa", split="train")
rows = [(r["question"], r["answer"], r["label"]) for r in d][:8000]
q = [a for a, _, _ in rows]
ans = [b for _, b, _ in rows]
y = np.array([c for _, _, c in rows])
uniq = {qq: i for i, qq in enumerate(dict.fromkeys(q))}
qid = np.array([uniq[qq] for qq in q])
print(f"n={len(y)} questions={len(uniq)} pos_rate={y.mean():.3f}", flush=True)

# ---------------- scoring ----------------
if os.path.exists(SCORE_CACHE):
    s = np.load(SCORE_CACHE)
else:
    s = np.full(len(y), np.nan)

todo = [i for i in range(len(y)) if not np.isfinite(s[i])]
if todo:
    import anthropic
    client = anthropic.Anthropic()

    PROMPT = (
        "Rate how well the candidate sentence answers the question, "
        "on a scale from 0 (completely irrelevant or does not answer) "
        "to 100 (directly and fully answers). Respond with a single "
        "integer and nothing else.\n\nQuestion: {q}\n\nCandidate sentence: {a}"
    )

    def score_one(i):
        for attempt in range(4):
            try:
                msg = client.messages.create(
                    model=MODEL, max_tokens=8, temperature=0,
                    messages=[{"role": "user",
                               "content": PROMPT.format(q=q[i], a=ans[i])}])
                m = re.search(r"-?\d+", msg.content[0].text)
                if m:
                    return i, float(m.group())
                return i, np.nan
            except Exception as e:
                if attempt == 3:
                    print(f"row {i} failed: {e}", flush=True)
                    return i, np.nan
                time.sleep(2 * (attempt + 1))

    print(f"scoring {len(todo)} rows with {MODEL}...", flush=True)
    done = 0
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for i, v in ex.map(score_one, todo):
            s[i] = v
            done += 1
            if done % 400 == 0:
                np.save(SCORE_CACHE, s)
                print(f"  {done}/{len(todo)}", flush=True)
    np.save(SCORE_CACHE, s)

ok = np.isfinite(s)
print(f"scored: {ok.sum()}/{len(s)}", flush=True)
if ok.sum() < len(s) * 0.98:
    print("WARNING: >2% failures; rerun to fill NaNs", flush=True)

# restrict to scored rows (keep question grouping intact)
q = [q[i] for i in range(len(y)) if ok[i]]
ans = [ans[i] for i in range(len(y)) if ok[i]]
y = y[ok]
qid = qid[ok]
s = s[ok]
print(f"analysis n={len(y)}; unique judge scores={len(np.unique(s))}, "
      f"mean={s.mean():.1f}, sd={s.std():.1f}", flush=True)

# ---------------- identical analysis pipeline ----------------
sys.path.insert(0, HERE)
from review_r4_experiments import run_setting, qa_features  # noqa: E402

phi, dense = qa_features(q, ans)
res = run_setting("wikiqa_haiku_judge", phi, dense, y, s, qid)
res["model"] = MODEL
res["n_scored"] = int(ok.sum())
res["raw_A_full"] = round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, s)), 4)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
