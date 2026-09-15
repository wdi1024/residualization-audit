"""Reviewer round-6, item E: does the 512-token scoring window hide the mutation?

The audit scores (problem statement, code) pairs at max_length=512. If truncation
cut the code before the mutated line, the reward model could not see the
correctness difference at all, and the paired format effects would be measured on
inputs that are identical across labels. This script measures, from the released
candidate cache and the audit's own tokenizer/limit:
  1. token-length distribution and truncation rate per cell;
  2. whether the truncated inputs of the correct and buggy twins remain distinct;
  3. the paired format effects recomputed on problems where nothing truncates.
Run: python3 truncation_check.py"""
import json, os
import numpy as np
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
rows = json.load(open(f"{CACHE}/mbpp_rows.json"))
s = np.load(f"{CACHE}/mbpp_rm_s.npy")
assert len(rows) == len(s) == 1412
tok = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-base")
MAXLEN = 512

cells = {}
for i, (q, code, y, pid, cell) in enumerate(rows):
    cells.setdefault(cell, []).append(i)
print("cells:", {k: len(v) for k, v in cells.items()})

lens = np.array([len(tok(q, code)["input_ids"]) for q, code, *_ in rows])
print(f"\ntoken length: median {np.median(lens):.0f}, p95 {np.percentile(lens,95):.0f}, max {lens.max()}")
for cell, idx in cells.items():
    l = lens[idx]
    print(f"  {cell:16s} median {np.median(l):6.0f}  max {l.max():5d}  truncated {(l>MAXLEN).sum():4d}/{len(l)}")
print(f"\noverall truncation rate: {(lens>MAXLEN).sum()}/{len(lens)}")

# 2. do correct/buggy twins stay distinct after truncation?
by_pid = {}
for i, (q, code, y, pid, cell) in enumerate(rows):
    by_pid.setdefault(pid, {})[cell] = i
def trunc_ids(i):
    q, code = rows[i][0], rows[i][1]
    return tuple(tok(q, code, truncation=True, max_length=MAXLEN)["input_ids"])
collapsed = {"terse": 0, "verbose": 0}
for pid, d in by_pid.items():
    for fmt in ("terse", "verbose"):
        a, b = d.get(f"correct-{fmt}"), d.get(f"buggy-{fmt}")
        if a is not None and b is not None and trunc_ids(a) == trunc_ids(b):
            collapsed[fmt] += 1
print(f"\ncorrect/buggy twins identical after truncation: "
      f"terse {collapsed['terse']}/{len(by_pid)}, verbose {collapsed['verbose']}/{len(by_pid)}")

# 3. paired format effects on problems where nothing truncates
def paired(pids, c1, c2):
    d = [s[by_pid[p][c1]] - s[by_pid[p][c2]] for p in pids
         if c1 in by_pid[p] and c2 in by_pid[p]]
    d = np.array(d); rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(1000)]
    return d.mean(), np.percentile(boot, [2.5, 97.5]), len(d)
all_pids = list(by_pid)
clean = [p for p in all_pids if all(lens[i] <= MAXLEN for i in by_pid[p].values())]
print(f"\nproblems with no truncated cell: {len(clean)}/{len(all_pids)}")
for name, c1, c2 in [("format @ correct", "correct-verbose", "correct-terse"),
                     ("format @ buggy", "buggy-verbose", "buggy-terse"),
                     ("construct @ terse", "correct-terse", "buggy-terse"),
                     ("construct @ verbose", "correct-verbose", "buggy-verbose")]:
    m_all, ci_all, n_all = paired(all_pids, c1, c2)
    m_cl, ci_cl, n_cl = paired(clean, c1, c2)
    print(f"  {name:20s} all n={n_all:3d} {m_all:+.3f} [{ci_all[0]:+.3f},{ci_all[1]:+.3f}] | "
          f"untruncated n={n_cl:3d} {m_cl:+.3f} [{ci_cl[0]:+.3f},{ci_cl[1]:+.3f}]")
