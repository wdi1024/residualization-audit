# Pre-registration: does format-effect attenuation generalize across edit templates?

Written 2026-09-08, **before** templates C and D are built or scored. Fixed at commit time;
nothing below is revised after results are seen.

## Motivation

The paper currently reports one held-out edit template (B). Fitting the residualizer on
template A and applying it unrefit to B **overshoots**: the raw format effect is
-0.159 [-0.249,-0.071] (correct) and -0.095 [-0.177,-0.013] (buggy), and residualization
moves it to +0.195 [+0.114,+0.275] and +0.259 [+0.184,+0.335]. The paper reads this as
"the attenuation is specific to the template it is fit on."

With n=1 held-out template that reading rests on a single comparison. Reviewers have twice
noted that the in-distribution success could be "fitting away one known rendering."

## Design

Two further comment/docstring-only templates, structurally unlike A and B:

- **C — inline comments, no docstring.** An end-of-line `#` comment on each non-trivial
  statement; no triple-quoted block anywhere. Drives the `has-docstring` indicator to 0
  while raising comment count and comment-character share.
- **D — header docstring, no inline comments.** A multi-line module-level docstring before
  the function and blank-line padding between logical blocks; no `#` comments at all.
  Drives comment count to 0 while raising blank-line count and total length.

C and D are chosen so that the nine-dimensional format feature vector moves in *different*
directions from A and B, not merely further along the same one.

Both are produced from the same 353 `correct-terse` / `buggy-terse` sources as B, and
**every label is re-verified by executing the problem's MBPP unit tests**; any problem whose
label does not survive the edit is dropped, exactly as in `template_holdout.py`.

Scoring: `OpenAssistant/reward-model-deberta-v3-base`, CPU, batch 32, max_length 512 —
the same scorer and settings as every other cell in the audit.

## Analyses (fixed in advance)

1. **Leave-one-template-out.** For each T in {A,B,C,D}: fit the cross-fitted ridge on the
   pooled rows of the other three, apply it unrefit to T, and report the within-problem
   paired format effect on T, raw vs. residualized, per correctness cell.
2. **Fit-on-A-only.** Apply the operator already reported in the paper, unrefit, to C and D.
   This is a direct replication of the existing B result at two more templates.
3. **Fit-on-A+B.** Does a second training template close the gap on C and D?

Primary endpoint for each: the paired raw-to-residualized change, problem-level bootstrap,
1,000 resamples, 95% CI, seed 0. Construct margins (correct - buggy) are reported beside
each as a design check, as in Table 1.

## Predictions, committed now

- **P1.** Under analysis 2, at least one of C, D shows the B pattern: the residualized
  format effect does not lie closer to zero than the raw effect (overshoot or sign flip).
- **P2.** Under analysis 1, the held-out paired change is smaller in magnitude than under
  analysis 2 for the same template — pooling templates helps, but does not close the gap.
- **P3.** Construct margins stay inside +/-0.02 of their raw values under every fit, since
  format remains balanced across labels by construction.

## Kill gate and what each outcome means

- **If P1 holds** (attenuation fails on further unseen templates): the paper's existing
  claim is strengthened from one held-out template to three, and the finding is promoted
  from a limitation to a central negative result — *format debiasing is rendering-specific*.
- **If P1 fails** (the A-fitted operator attenuates C and D): the B overshoot is a property
  of template B, not of the operator. The paper's sentence "the attenuation is specific to
  the template it is fit on" is **too strong and must be narrowed**, and the B result is
  reported as one template on which the operator fails rather than as a general limit.
- **If analysis 3 closes the gap on both C and D**: the claim becomes "one training template
  is not enough", which is a different and weaker statement than the one now in the paper.

Any of the three outcomes changes text that is currently in the manuscript. A headline
change requires author approval before it is written in.

## Cost

- Building C and D: no compute.
- Label re-verification: 4 x 706 subprocess test runs, 10s timeout each, ~2-4 min total.
- Scoring: 1,412 new candidates on CPU (706 per template), ~2-4 min total.
- Analyses: seconds.
- **No GPU, no API calls, no paid resources.** Wall clock ~15-25 min end to end.

## Output

`scripts/template_generalization.py`, results to
`notes/template_generalization.json`, scored cells cached as
`cache/mbpp_rm_templateC_s.npy` and `cache/mbpp_rm_templateD_s.npy`.
