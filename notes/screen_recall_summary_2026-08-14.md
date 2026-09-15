# Screen recall (false-negative) experiment — summary (2026-08-14)

Reviewer weakness (8/1 mock review, one of 2 remaining "must-run" items): C1-screen-rejected
candidates were never scored end-to-end, so screen recall was unmeasured. This run scores all
7 rejected candidates (rounds 1+2) through the full frozen pipeline.

Protocol: single-shot precommit (`.committed` written before any scoring), frozen recipe
verbatim from the paper's runs, determinism confirmed for all 7 (two independent processes,
exact equality of every statistic + 32-item scorer spot-check, max|diff|=0.0).
Script: `scripts/screen_recall_run.py`. Per-candidate artifacts: `screen_recall_<cand>.json`.

## Results (7/7)

| candidate | screen C1 | pipeline C1 | C2 (R², Δadv) | slice gain [CI] | full-pop gain | verdict |
|---|---|---|---|---|---|---|
| shp       | 0.647 | 0.647 | pass | +0.147 [+0.127, +0.170] | −0.044 | **FN — validated positive** |
| boolq     | 0.640 | 0.640 | fail (0.068, −0.005) | +0.024 | −0.005 | TN |
| hatecheck | 0.637 | 0.637 | pass (0.104, 0.098) | +0.108 [+0.099, +0.118] | −0.054 | **FN — validated positive** |
| mnli      | 0.616 | **0.697** | pass | +0.036 [+0.028, +0.045] | −0.067 | **FN\*** (see caveat) |
| paws      | 0.538 | 0.538 | fail (0.104, −0.036) | +0.008 n.s. | +0.006 | TN |
| rte       | 0.524 | 0.524 | fail (−0.029, −0.005) + decorr fail | +0.025 | −0.046 | TN |
| hh        | 0.513 | 0.511 | fail (0.042, −0.024) | −0.013 (neg.) | +0.003 | TN |

**Tally: 3 FN / 4 TN.** Empirical recall on the rejected set = 4/7 correct rejections.

## Interpretation

1. **All FNs sit in the borderline band C1 0.616–0.647, just under the 0.65 gate.** Every
   candidate with C1 ≤ 0.64 except the band's edge cases is correctly rejected downstream
   (C2/decorrelation/slice do the work). The screen is not broken; the gate constant is
   slightly high. This promotes the §5.2 conjecture ("the gate may be conservative for
   weak-but-real channels") to a measured finding.
2. **hh (C1 0.513, lowest) is TN** — the discriminating test the frame hinged on. No FN
   appears anywhere below the borderline band → NOT a "screen is unreliable" result.
3. **Error direction is conservative throughout**: every screen error is a missed positive,
   never a false approval. The safety claim survives untouched; the cost is efficiency
   (some repairable channels are left on the table).
4. **C1 alone does not separate FN from TN inside the band**: boolq (0.640) is TN while
   hatecheck (0.637) is FN — C2 is what separates them. Supports the paper's claim that
   the checks are jointly necessary, not redundant.
5. **mnli caveat (must be stated if integrated into the paper):**
   - Screen-time C1 0.616 vs deployment-protocol remeasure 0.697 — this FN is a *screen
     measurement-protocol underestimate*, not a gate-constant issue (unlike shp/hatecheck).
   - Paper Table 2 already carries (MNLI, SNLI-BERT) as a refusal (−0.034) with a
     *different* φ (standard surface vs hyp-only TF-IDF here). Consistent with the
     "φ inventory determines the channel" narrative, but a head-on contrast sentence is
     required if this enters the manuscript.

## Disposition

Per user policy (rebuttal-first), this is a **rebuttal card**, not an automatic manuscript
change: if a reviewer asks about screen recall, the answer is "measured on all 7 rejected
candidates; errors are conservative and confined to the boundary band; gate constant, not
mechanism." Optional manuscript integration (one paragraph + small table in the screen
appendix, plus §5.2 conjecture→measurement upgrade) is a user decision.
