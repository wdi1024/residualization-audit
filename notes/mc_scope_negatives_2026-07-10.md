# HellaSwag + SWAG: two more predicted negatives (2026-07-10)

`scripts/mc_positive_run.py` — MC items flattened to (context, ending, y)
pairs; s = distilbert-MNLI entailment (off-dist); phi = ending-only TF-IDF;
GroupKFold(item) CV; same operator/gates as SNLI/SICK.

## Results (n=2400 pairs / 600 items each)

| dataset | C1 ending-only AUC | C2 raw AUC | adv-slice gain |
|---|---|---|---|
| HellaSwag | 0.590 (< 0.65 FAIL) | 0.573 (model ~broken) | −0.016 [−0.066,+0.041] n.s. |
| SWAG | 0.563 (FAIL) | 0.563 (~broken) | −0.013 [−0.066,+0.037] n.s. |

Both fail BOTH gates → the precondition predicts no positive, and there is
none. Two clean scope rows, zero surprises.

## Reading

1. The literature's "strong ending-only artifact" in SWAG/HellaSwag is a
   *BERT-finetuned* partial-input result; **linear TF-IDF does not recover
   it** (C1 is specifically LINEAR recoverability — what the linear repair
   operator needs). Same family as the HANS "structural/nonlinear" row.
2. distilbert-MNLI entailment does not transfer to commonsense-continuation
   ranking (raw ≈ 0.57): ANLI-type "model broken" case — nothing to repair.

## Updated precondition table (9 settings)

2 positives (SNLI, SICK — both gates pass, SICK prospectively predicted) and
7 negatives (MNLI-off-dist, HANS, toxicity, QQP, ANLI, HellaSwag, SWAG — each
fails ≥1 gate, all correctly predicted). The validity condition is now
9-for-9 as a predictor. Optional-strengthening budget is spent; paper 6 is
writing-only.
