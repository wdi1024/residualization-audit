# Certificate–validity trade-off (round-4 must-fix #1, cross-session verification)

Two independent implementations (tmux: cert_reverify.py; discord-session:
scripts/certificate_exact{,2}.py) agree on the channel correlations after
ridge repair: |corr(s_rep, a_hat)| = 0.014 (SNLI) / 0.015 (SICK) / 0.031
(WikiQA), meeting the kappa<=0.05 closure criterion.

Decision-level probe AUC (predicting z from the score, App F protocol):

| residualizer | SNLI | SICK | WikiQA |
|---|---|---|---|
| ridge_cv (paper) | 0.451 | 0.648 | 0.559 |
| ols_cv | 0.492 | 0.540 | 0.523 |
| ols_full (in-sample) | 0.489 [.465,.513] | 0.481 [.449,.514] | 0.494 [.465,.524] |

The exact OLS projection closes even the decision floor — but by DESTRUCTION:
construct alignment of the OLS-repaired score (full set):
  SNLI ridge 0.880 / ols_cv 0.715 / ols_full 0.534   (raw 0.958)
  SICK ridge 0.906 / ols_cv 0.777 / ols_full 0.664   (raw 0.967)
(p=8000 TF-IDF >> n=2400: unregularized fit interpolates, s_rep -> noise.)

Paper text (App F "decision-level diagnostic" + §2.2) updated to state the
trade-off explicitly; the shrinkage operator is the deliberate choice.
R2(phi->s_rep) <= 0 for all variants (guaranteed-class metric closed everywhere).
Full JSON: notes/certificate_exact_2026-07-11.json.
