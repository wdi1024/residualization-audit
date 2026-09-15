# ASNQ candidate-protocol sensitivity sweep — summary (2026-08-14)

Mock-review (8/1) remaining item #2. Devil's-advocate concern: the resampled candidate
protocol (all-pos + ≤19 neg/question, C1 0.643→0.747) was chosen *after* observing the
natural-sampling C1, so the gate pass could be an adaptive knife-edge. Required sweep:
candidates/query {5,10,20,50,all} × negative-sampling seeds × natural-vs-balanced base
rate, reporting joint movement of C1, C2, and repair gain.

Protocol: precommit before scoring; recipe verbatim (dense-screen C1 on KFold(5,s,0) —
comparable to the 0.643/0.747 anchors; question-grouped GroupKFold(5) pipeline; MS MARCO
MiniLM-L6 CE scorer; 1000 question-clustered bootstrap). 101,963 new (q, sent) pairs
scored once into a global cache. Determinism PASS (all statistics identical across two
processes; spot-check max|diff| < 1e-6). Script: `scripts/asnq_candidate_sensitivity.py`;
data: `notes/asnq_candidate_sensitivity.json`.

## Anchors reproduce exactly
- `natural` (screen-verbatim, incl. no-positive questions): C1 **0.643** (= July value).
- `cached20` (the paper's exact resampled 8k rows + cached scores): C1 **0.747**
  (= July value); pipeline matches the paper's grouped rerun closely
  (R² 0.370 vs 0.377, Δadv 0.360 vs 0.356, slice +0.198 vs +0.200, full −0.079 vs −0.082,
  cert 0.145 vs 0.139 — residual diffs are group-ordering/bootstrap noise).

## Findings

1. **C1 pass is a plateau, not a knife-edge.** Across candidates/query 5→50, all three
   seeds, and balanced base rate: C1 ∈ **0.735–0.767** — comfortably above the 0.65 gate
   everywhere. Even `all` (every candidate, natural base rate *within answerable
   questions*) gives 0.694–0.845. The ONLY sub-gate config is `natural` (0.643), which
   differs by including questions with no positive at all. So the 0.643→0.747 jump is not
   an artifact of tuning k: **any within-question candidate-list protocol passes**; the
   natural-sampling failure is driven by no-positive questions diluting the within-question
   structure, which is exactly the deployment-distribution argument the paper already makes.
2. **Seed sensitivity is small**: C1 spread ≤0.02 within each k; slice gain spread ≤0.03.
3. **C2 and repair direction are protocol-invariant**: R² 0.09–0.51 and Δadv 0.25–0.44
   pass the gates in every config; slice gain is significantly positive everywhere
   (+0.10 to +0.21); full-population gain is negative everywhere (−0.02 to −0.16, and
   grows with base-rate balance) — the paper's joint slice-vs-full pattern is stable
   under every candidate protocol.
4. **The known decorrelation failure is systematic, monotone in list length — and the
   paper's choice sat on the honest side of it.** |corr(s_rep, index)| after repair:
   k5 0.018–0.057, balanced 0.024–0.042 (pass/marginal) → k10 ~0.07–0.08 → k20 0.11–0.15
   → k50 ~0.20 → all 0.20–0.34 (fail). Had the protocol been chosen adaptively to make
   ASNQ look best, k≈5 or balanced would have produced a fully validated positive; the
   paper's k≈20 instead recorded the decorrelation failure it reports. The adaptivity
   charge therefore runs in the paper's favor.

## Disposition

Rebuttal card (per user policy), alongside the screen-recall card. If integrated: one
paragraph + small table in the ASNQ appendix; the decorrelation-monotonicity point (4) is
also usable as an independent observation about base-rate skew inflating residual
correlation. n-caveat: every config is capped at 8000 rows (screen verbatim), so
candidate-count effects are partly question-count effects (more candidates/query = fewer
questions at fixed n); state this if published.
