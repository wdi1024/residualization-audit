"""CPU rescoring of the scorer-replication candidates that had no cached scores:
ASNQ x {stsb, qnli, electra}, TriviaQA x {stsb, qnli}. Same recipe as the
original scorer scripts (CrossEncoder, max_length=256). Scores cached to
../cache/ so downstream grouped analysis is reproducible without inference.
"""
import json, os, time
import numpy as np
from sentence_transformers import CrossEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")

MODELS = {
    "stsb": "cross-encoder/stsb-roberta-base",
    "qnli": "cross-encoder/qnli-distilroberta-base",
    "electra": "cross-encoder/ms-marco-electra-base",
}

def load_asnq():
    rows = json.load(open(f"{CACHE}/asnq_rows_8000.json"))
    return [(a, b) for a, b, _ in rows]

def load_trivia():
    rows = np.load(f"{CACHE}/triviaqa_rows.npy", allow_pickle=True)
    return [(r[0], r[1]) for r in rows]

JOBS = [("asnq", load_asnq, ["stsb", "qnli", "electra"], "asnq_{m}_s_8000.npy"),
        ("triviaqa", load_trivia, ["stsb", "qnli"], "triviaqa_{m}_s.npy")]

for name, loader, models, pattern in JOBS:
    pairs = loader()
    for m in models:
        out = os.path.join(CACHE, pattern.format(m=m))
        if os.path.exists(out):
            print(f"[skip] {out} exists", flush=True)
            continue
        t0 = time.time()
        ce = CrossEncoder(MODELS[m], max_length=256)
        s = ce.predict(pairs, batch_size=64, show_progress_bar=False)
        np.save(out, np.asarray(s, dtype=np.float32))
        print(f"[done] {name}/{m}: n={len(pairs)} in {time.time()-t0:.0f}s -> {out}", flush=True)
print("all rescoring done", flush=True)
