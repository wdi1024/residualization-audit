# SICK: the second positive — and a PROSPECTIVE validation of the precondition (2026-07-10)

`scripts/sick_positive_run.py` (data via datasets-server mirror `yangwang825/sick`,
test split; the deprecated-loader blocker from 07-09 is gone). Same recipe as the
SNLI positive: distilbert-MNLI scored OFF-distribution, phi = hypothesis-only
TF-IDF(8k, 1-2gram), repair = CV-Ridge residualization, seed 0.
Label semantics recovered by model-agreement probe (0=entail, 2=contradiction).

## Result (n=2116 binary ent-vs-contra)

- C1 artifact strength (hyp-only AUC): **0.727** (gate ~0.65 → PASS)
- C2 corruption: raw full AUC 0.967 BUT adversarial slice 0.765 → the score
  visibly rides the artifact where it misleads (PASS)
- **Adversarial slice: raw 0.765 → repaired 0.964, gain +0.199 [CI +0.018, +0.450]**
- Full AUC trade-off: 0.967 → 0.906 (same honest shape as SNLI)

## Why this is the strongest result in the paper

The precondition (C1+C2) was written down on 07-09 BEFORE this run, and the
07-09 note explicitly predicted: "a model trained on dataset A applied to a
different dataset B with a genuine partial-input artifact — candidate: SICK —
the positive should appear." It did. This upgrades the validity condition from
a post-hoc pattern over 6 settings to a **prospectively validated predictor**
(7 settings: 2 positive exactly where predicted, 5 negative exactly where
predicted).

Updated table:

| setting                      | artifact AUC | corrupted? | adv-slice repair |
|------------------------------|--------------|------------|------------------|
| SNLI (distilbert-MNLI)       | 0.750        | yes        | 0.928→0.967 ✅   |
| **SICK (distilbert-MNLI)**   | **0.727**    | **yes**    | **0.765→0.964 ✅ (prospective)** |
| MNLI (SNLI-bert off-dist)    | 0.617        | no (0.923) | −0.034 ✗         |
| HANS (distilbert-MNLI)       | R²≈0 struct. | nonlinear  | mixed ✗          |
| toxicity CivilComments       | 0.557        | no (0.953) | +0.008 n.s. ✗    |
| QQP (in-dist BERT-QQP)       | 0.636        | no (0.993) | −0.045 ✗         |
| ANLI (distilbert-MNLI)       | 0.556        | broken     | −0.043 ✗         |

## Paper status after this run

Skeleton complete: validity condition + repair operator + synthetic ρ-curve +
RLC unrepairable anchor + **2 real positives** + 5 scope negatives with
predictive precondition. **Remaining: writing only** (LaTeX draft + figures).
SICK gain CI is wide ([+0.018,+0.450], n_adv=455) — report honestly; SNLI is
the tighter headline, SICK is the prospective-validation story.
