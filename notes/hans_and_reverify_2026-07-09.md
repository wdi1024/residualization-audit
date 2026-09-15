# HANS run + SNLI headline re-verification (2026-07-09)

## Bug found & fixed (affects HANS only, not SNLI)
`cross_val_predict(..., cv=5)` uses **unshuffled KFold**. HANS eval file is ordered by
heuristic block (constituent → lexical_overlap → subsequence), so contiguous folds split
by block → cross-fit trained on some heuristics, predicted on a disjoint one. The residual
then removed **block-specific offsets** (leakage), inflating the "positive" to a spurious
global +0.08. Fixed with `KFold(n_splits=5, shuffle=True, random_state=0)` everywhere.
SNLI val is NOT block-ordered, so its headline was unaffected (confirmed below).

## HANS, corrected (distilbert-base-uncased-mnli, n=8000 stratified, dense φ = overlap+lengths, shuffled CV)
- global AUC(y=entail): 0.640 → 0.603 = **−0.037** [−0.050, −0.023]  (repair HURTS globally)
- per-heuristic AUC raw→repaired, gain[95%CI]:
  - lexical_overlap : 0.592 → 0.697  **+0.104** [+0.084, +0.124]  ✅ (only heuristic where repair helps)
  - constituent     : 0.714 → 0.650  −0.064 [−0.085, −0.044]
  - subsequence     : 0.617 → 0.417  −0.200 [−0.225, −0.175]
- surface-R²(φ→s) ≈ **0.02**  → distilbert's HANS score is NOT a linear length/overlap artifact.
- φ ablation (shuffled): dense (overlap+len) drives the lexical_overlap effect; hypothesis
  TF-IDF-only gives global +0.000 (inert, not label-fitting).

### Reading (a *scope-condition* result, not a 2nd global positive)
HANS sharpens the method's scope: **repair helps iff the artifact channel is (a) linearly
recoverable AND (b) misaligned with the construct.** lexical_overlap satisfies both →
positive. constituent/subsequence: the surface cue is construct-*aligned* (or the failure is
nonlinear/structural, R²≈0) → residualizing removes signal → repair hurts. This is the
validity condition demonstrated *within a single canonical benchmark*, complementing the
synthetic ρ-curve and the RLC unrepairable anchor. It is NOT a clean global positive; do not
sell it as one.

## SNLI headline RE-VERIFIED with shuffled CV — SURVIVES
`scripts/nli_positive_run.py` (distilbert-mnli, SNLI val binary entail-vs-contra, n=2400,
φ = hypothesis-only TF-IDF, cross-fit Ridge residual, shuffled KFold):
- hypothesis-only AUC (artifact strength): **0.708**
- artifact-adversarial slice (hyp-only WRONG, n=863): raw **0.927 → repaired 0.970** ✅ RISES
- full set: 0.958 → 0.880 (falls; artifact construct-aligned on the full distribution)
Prev (unshuffled) was 0.928→0.967 — essentially identical, so the SNLI positive is robust and
was never affected by the fold bug. **SNLI remains the paper's real positive headline.**

## Net three-part story (unchanged in shape, sharpened)
1. POSITIVE — SNLI hyp-only adversarial slice: 0.927 → 0.970. Repair works when the artifact
   is linear + misleading.
2. VALIDITY CONDITION / SCOPE — synthetic ρ-curve + HANS per-heuristic (helps on
   lexical_overlap, hurts where artifact is construct-aligned/nonlinear). Repair is not free;
   it is licensed only under the orthogonality+linearity condition.
3. UNREPAIRABLE ANCHOR — RLC XSTest-450: raw proxy 0.906 / construct 0.278 → repaired
   0.771 / 0.372; the anchor score is artifact-dominated and construct-collinear → can't recover.

## Honest caveat for the writeup
The "canonical benchmark global positive" we hoped HANS would give does NOT exist for a linear
repair (distilbert's HANS errors are structural, surface-R²≈0). Present HANS as the scope/
validity demonstration, keep SNLI as the positive. Optionally test a length-biased model or a
task with a *known linear* artifact for a second clean positive.
