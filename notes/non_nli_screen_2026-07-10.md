# Non-NLI positive hunt, round 2: C1 screen (2026-07-10)

`scripts/non_nli_screen.py` — artifact-only AUC, no neural compute, gate ~0.65.

| candidate | artifact | C1 AUC | verdict |
|---|---|---|---|
| **WikiQA (n=8000, pos 4.9%)** | overlap+lengths (dense) | **0.747** | ✅ PASS — strongest linear channel found anywhere, incl. SNLI 0.708 |
| WikiQA | answer-only TF-IDF | 0.677 | pass (secondary) |
| SHP preference (n=4000) | length-only | 0.647 | ✗ borderline fail (MNLI-territory) |
| BoolQ (n=6000) | question-only TF-IDF | 0.640 | ✗ fail |
| HateCheck (n=3728) | identity+length | 0.637 | ✗ fail |
| hh-rlhf (n=4000) | length-only | 0.513 | ✗ chance |

Notes:
- The famous RM "length bias" is real but WEAK as a linear channel on SHP
  (0.647) and absent on hh-rlhf (0.513) — same lesson as HellaSwag/SWAG: the
  literature's artifact ≠ a linearly recoverable artifact.
- Negatives keep clustering in 0.51–0.65, right below the gate → reinforces
  the threshold-sensitivity story (App. D gap is (0.64, 0.71)).
- WikiQA picked for full run: MS MARCO cross-encoder (ms-marco-MiniLM-L-6-v2)
  as off-distribution scorer, y = crowdsourced answer label (independent),
  rate-calibrated adversarial slice (base rate 4.9%).
  → `scripts/wikiqa_positive_run.py`, results in `wikiqa_result_2026-07-10.json`.
