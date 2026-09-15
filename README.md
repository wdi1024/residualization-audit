# When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits

Code, pre-registration artifacts, and cached scores for the paper
**"When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits"**
(Dong Ho Kang, HyeonJeong Cha, Daein Weon).

The paper asks what a post-hoc residualization of an evaluation score establishes once it has been
applied: what designed interventions can show, what observational slice gains actually support, and
where diagnostic success stops implying validity.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Per-setting analysis scripts (designed audit, observational settings, synthetics) |
| `grouped_rerun/` | Scripts for the question-grouped protocol used in the headline tables |
| `notes/` | Pre-registration artifacts and per-setting run records |
| `cache/` | Cached scorer outputs, so the analyses can be rerun without re-scoring |

Every script named in the paper lives in `scripts/` or `grouped_rerun/`. The paper itself is on
arXiv; this repository holds only the material behind it.

## Pre-registration artifacts

Files ending in `.committed` under `notes/` are the decisions recorded **before** the corresponding
scorer was run. They are the record behind the chronology stated in the paper: gate constants fixed
on the first five settings, later settings screened before scoring, and two held-out replications
frozen end to end before any score existed. These files are not third-party timestamped, and the
paper says so.

## Reproducing

The analyses run CPU-only on the cached scores and are seeded. Scripts read API credentials from the
environment where an LLM judge is involved; no keys are stored in this repository.

```bash
python scripts/<script>.py
```

## Citation

A BibTeX entry will be added when the preprint is posted.
