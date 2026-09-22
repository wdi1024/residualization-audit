# When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits

Code, pre-registration artifacts, and cached scores for the paper
**"When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits"**
by Daein Weon and Dong Ho Kang.

📄 Paper: [arXiv:2609.24194](https://arxiv.org/abs/2609.24194)

The paper asks what a post-hoc residualization of an evaluation score establishes once it has been
applied: what designed interventions can show, what observational slice gains actually support, and
where diagnostic success stops implying validity. It measures what adjustment buys in three
regimes — designed interventions, observational NLI and QA settings, and an entangled semi-synthetic
probe — and this repository holds the material behind each of those verdicts.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9 on CPU. `torch`, `transformers` and `sentence-transformers` are needed only
by the scripts that re-score with a model (10 of the 40 scripts named in the paper); the rest read
the cached scores and need only `numpy`, `scipy`, `scikit-learn` and `datasets`.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Per-setting analysis scripts: designed audit, observational settings, synthetics |
| `grouped_rerun/` | Scripts for the question-grouped protocol used in the headline tables |
| `notes/` | Pre-registration artifacts (`*.committed`) and per-setting run records |
| `cache/` | Cached scorer outputs, so the analyses can be rerun without re-scoring |
| `*.json` (repo root) | Intermediate results written by the scripts, read back by later ones |

Scripts resolve their inputs relative to their own file location, not to the working directory, so
they can be run from anywhere — but the five JSON files at the repository root and the four
directories above must stay where they are.

## Reproducing

The analyses run CPU-only on the cached scores and are seeded:

```bash
python grouped_rerun/repeat_fit_stability.py    # e.g. the fit-stability check of Appendix P
```

Scripts read API credentials from the environment where an LLM judge is involved
(`ANTHROPIC_API_KEY`); no keys are stored in this repository.

### Which script produces which result

Every script named in the paper is listed here.

| Paper location | Scripts |
|---|---|
| §3 A Designed Intervention Audit: a Reward Model on Code | `rm_code_paired_change.py` |
| §4.2 The same-label settings, and the dose–response underneath them | `judge_gain_ci.py` |
| §5.2 Measured costs outside the slice | `grouped_full.py` |
| App. G A Probe of the Entangled Regime by Assignment | `bridge_verdicts.py`, `bridge_win_cis.py`, `rescore_mbpp_judges.py` |
| App. L Candidate Selection by the Slice-Eligibility Check | `screen_recall_run.py` |
| App. M The Reward-Model Code Audit: Construction and Protocol | `rm_code_audit.py`, `rm_code_build.py`, `rm_code_dose.py`, `rm_code_extra.py`, `paired_g_contrast.py`, `length_only_baseline.py`, `nonlinear_length_baseline.py`, `template_generalization.py`, `template_holdout.py`, `truncation_check.py` |
| App. N Identifiability | `g_identification_counterexample.py` |
| App. O Screen Operationalization and Deployment-Rule Negatives | `obs_rule_search.py`, `rule_search_rescaled.py`, `deployment_grouped.py`, `obs_risk_flag.py`, `risk_flag_switch.py`, `overlap_intervention.py`, `overlap_edit_llm_check.py`, `review_r2_experiments.py` |
| App. P The Question-Grouped Protocol, Score Representation, and Query-Level Ranking | `grouped_full.py`, `scorer_grouped.py`, `rescore_scorers.py`, `scale_sensitivity.py`, `repeat_fit_stability.py` |
| App. Q.3 Operator robustness | `review_r4_experiments.py` |
| App. R Scope of the Decorrelation Diagnostic | `review_r3_experiments.py` |
| App. S The Cross-Fitted Conditional-Quantile Eraser | `quantile_eraser.py` |
| App. T Slice Definition: Selection-Bias Checks | `placebo_slice_test.py`, `dual_measurement_check.py`, `squad_dual_eval.py` |
| App. U Slices Drawn With Other Features | `cross_family_slice.py`, `feature_split_slice.py` |
| App. V Synthetic Stress Tests | `multi_artifact_noise_synthetic.py`, `t3_verdict.py` |

The remaining scripts in `scripts/` and `grouped_rerun/` are the per-setting scoring and screening
runs that produced the caches these analyses read.

## Pre-registration artifacts

Files ending in `.committed` under `notes/` are the decisions recorded **before** the corresponding
scorer was run. They are the record behind the chronology stated in the paper: gate constants fixed
on the first five settings, later settings screened before scoring, and two held-out replications
frozen end to end before any score existed. These files are not third-party timestamped, and the
paper says so.

## Citation

```bibtex
@misc{weon2026residualization,
  title         = {When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits},
  author        = {Weon, Daein and Kang, Dong Ho},
  year          = {2026},
  eprint        = {2609.24194},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.24194}
}
```

## License

MIT — see [LICENSE](LICENSE).
