"""#5 (TMLR review): show the C1 gate (0.65) is NOT tuned to the observed settings. For each
setting, derive the threshold ONLY from the OTHER settings (midpoint of the gap between the highest
negative C1 and the lowest positive C1 among them), then classify the held-out setting. If every
held-out setting is correctly classified by a threshold it did not participate in choosing, the
verdicts are a property of the C1 gap, not of a hand-picked constant. Pure re-analysis, no compute.
"""
import numpy as np, json, os

# per-setting linear artifact-only AUC (C1) and ground-truth repairability, from the notes.
# scalar-C1 settings only (HANS enters via surface-R^2; the routing anchor is a collinearity/C2
# case, not a C1 case -- both are excluded from the C1 gap analysis by construction).
SETTINGS = {
    # positives (repairable)
    "SNLI": (0.708, 1), "SICK": (0.727, 1), "WikiQA": (0.747, 1),
    # negatives (refused), scored
    "QQP": (0.636, 0), "MNLI": (0.617, 0), "Toxicity": (0.557, 0),
    "ANLI": (0.556, 0), "HellaSwag": (0.590, 0), "SWAG": (0.563, 0),
    # negatives rejected at the pre-scoring C1 screen (highest negatives -> hardest)
    "SHP": (0.647, 0), "BoolQ": (0.640, 0), "HateCheck": (0.637, 0),
}

names = list(SETTINGS)
correct = 0; rows = []
for held in names:
    others = {k: v for k, v in SETTINGS.items() if k != held}
    pos = [c for c, y in others.values() if y == 1]
    neg = [c for c, y in others.values() if y == 0]
    thr = (max(neg) + min(pos)) / 2                 # gap midpoint from the OTHER settings only
    c1, y = SETTINGS[held]
    pred = int(c1 >= thr)
    ok = pred == y
    correct += ok
    rows.append({"held_out": held, "C1": c1, "threshold_from_others": round(thr, 4),
                 "predicted": pred, "true": y, "correct": ok})

allc1 = [c for c, _ in SETTINGS.values()]
pos_min = min(c for c, y in SETTINGS.values() if y == 1)
neg_max = max(c for c, y in SETTINGS.values() if y == 0)
print(f"leave-one-setting-out over {len(names)} scalar-C1 settings")
print(f"{'held-out':11}{'C1':>7}{'thr(others)':>13}{'pred':>6}{'true':>6}{'ok':>5}")
for r in rows:
    print(f"{r['held_out']:11}{r['C1']:>7.3f}{r['threshold_from_others']:>13.3f}"
          f"{r['predicted']:>6}{r['true']:>6}{str(r['correct']):>5}")
print(f"\ncorrect: {correct}/{len(names)}")
print(f"observed gap: highest negative C1 = {neg_max:.3f} (SHP), lowest positive C1 = {pos_min:.3f} "
      f"(SNLI); margin = {pos_min - neg_max:.3f}. Every held-out setting is classified correctly by a "
      f"threshold chosen without it -> the gate is a property of the gap, not a tuned constant.")
json.dump({"rows": rows, "correct": correct, "n": len(names),
           "pos_min_C1": pos_min, "neg_max_C1": neg_max, "margin": pos_min - neg_max},
          open(os.path.join(os.path.dirname(__file__), "..", "notes",
                            "leave_one_setting_out_2026-07-11.json"), "w"), indent=1)
