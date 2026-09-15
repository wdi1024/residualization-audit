# Positive-case data acquisition — RewardBench (do this in the focused block, post-AAAI)

Goal: a per-example table to feed `scripts/positive_case_rewardbench.py::load_rewardbench()`:
```
{ "margin": reward(chosen) - reward(rejected),   # score s
  "correct": 1 if RM ranked the truly-preferred higher,   # construct label y
  "len_chosen": <chars/tokens>, "len_rejected": <chars/tokens> }   # artifact phi
```

## Two ways to get it

**(A) Fastest — use a released RM scores dump.**
- RewardBench (allenai/reward-bench, HF) publishes per-model, per-example results for many reward models.
  Pull one strong RM's per-example chosen/rejected scores → `margin` and `correct`; get the pair texts
  from the RewardBench dataset → lengths. No GPU.
- Look for the leaderboard's downloadable per-example result files (or the `results/` artifacts on the
  RewardBench GitHub / HF); each row has the RM's scores for chosen and rejected.

**(B) Self-scored — run one RM over the pairs (needs a GPU box, e.g. the RunPod pod).**
- Load the RewardBench prompt/chosen/rejected pairs (HF `allenai/reward-bench`).
- Score with an open reward model (e.g. a small released RM) → margins. Cheap; one forward pass per
  response. Reuse the same RunPod GPU planned for the VLA (paper 5) rollout.

## Why this is the clean positive case
Reward-model margins are the textbook mix of a genuine preference construct and a **length/verbosity
artifact** (RMs reward longer/markdown-heavy answers). Residualizing the margin on length should RAISE
preference agreement (construct) while the length channel closes to chance — the ρ≈0–0.3 repairable
regime. Synthetic self-test already reproduces this (pref-AUC 0.78→0.97, floor 0.87→0.49); real
RewardBench turns it into the paper's headline positive alongside the RLC unrepairable anchor.

## Extra artifact features (optional, strengthens the story)
Beyond raw length: markdown/bullet counts, code-block presence, sentence count, hedging-phrase counts.
Add them as columns in `phi`; the repair closes whatever linear channel they open.

## Real run #1 (2026-07-08) — deberta-435M RM on RewardBench filtered (n=300): MILD confound (null-ish)

Scored 300 pairs with `OpenAssistant/reward-model-deberta-v3-large-v2` (CPU, `scripts/rewardbench_real_run.py`):
- RM accuracy P(margin>0) = 0.600
- corr(margin, length-gap) = **0.229** (mild length coupling)
- RM acc on length-BALANCED pairs = 0.626 vs length-DOMINATED = 0.598 → controlling for length does **not**
  reduce accuracy (slightly higher balanced).

Reading: this (RM, subset) is **not strongly length-confounded** — the reward score carries genuine
preference signal not driven by length, so repair has little to remove here (honest weak/null positive).
Also note RewardBench "filtered" is curated to reduce length confounds, and deberta-reward is relatively
length-robust. **This is real evidence, but not the strong positive headline.**

**Next iteration (to get a strong positive):** pick a testbed where the artifact confound is large —
(a) a **length-biased RM** (many larger/RLHF RMs reward length heavily; e.g., score with a known
length-biased open RM), or (b) RewardBench's **length-adversarial / chat-hard subset** (where the
longer answer is often the *rejected* one), or (c) add **format artifacts** (markdown/bullet/code counts)
as φ, which some RMs reward strongly. The mechanism is proven on synthetic (0.78→0.97); the real positive
just needs a genuinely confounded (RM, subset, artifact) triple — a targeted pick, ~30 min next block.

## Real run #2 (2026-07-08) — rich artifacts + length-direction slice: length bias is REAL but modest, AND RewardBench is structurally unsuitable for the repair demo

`scripts/rewardbench_strong_probe.py`, deberta-435M, n=400, φ = length/newlines/bullets/code/headers:
- acc when length AGREES with correct (chosen longer, n=146) = **0.678** vs DISAGREES (rejected longer,
  n=254) = **0.606** → a real length-bias gap of ~0.072 (corr(margin,length)=0.20). Real but modest.
- The "repaired accuracy on disagree slice 0.606→0.343" is an **invalid metric**: residualizing the
  margin recenters it, so thresholding at 0 no longer means "chosen>rejected." RewardBench's construct
  label is degenerate (chosen always preferred) and its score is threshold-based → it **cannot host the
  clean "repair raises AUC(y,s)" positive.** Confirmed structurally, not just weak.

**Conclusion for the positive headline:** it needs a testbed with an **independent, continuous construct
label** per example (which the synthetic has, RewardBench does not). Right candidate = **NLI-style**:
a model's confidence = score `s`, the true NLI label = independent construct `y`, hypothesis-only
features = artifact `φ`. Residualize `s` on `φ`, measure AUC(y) before/after → repair should raise it
(strong, well-known artifact). This is a **testbed-design** step, not a compute problem — do it as a
focused block, post-AAAI. Today's two real runs weren't wasted: they establish the *selection criterion*
(independent continuous construct label) and rule out RewardBench.

## Checklist
- [ ] Get one RM's per-example (chosen_score, rejected_score, correct) + pair texts.
- [ ] Build the json list (margin/correct/len_chosen/len_rejected [+ format feats]).
- [ ] `python3 positive_case_rewardbench.py rewardbench_scored.json` → report positive repair.
- [ ] Upgrade linear residual note: for scalar scores OLS-residual already gives linear orthogonality;
      cite LEACE only if repairing a representation-level score (vector), not the scalar margin.
