"""Reviewer round-6, item F: is the matched-format preference change really
'only cross-fitting noise'?

Within a format cell the paired construct margin changes by exactly the label
contrast of the subtracted component, g(phi_correct) - g(phi_buggy). If the two
twins had identical phi the contrast would be exactly zero and cross-fitting
could not move the within-pair ordering at all; they do not, so this script
measures the actual distribution instead of asserting it.
Run: python3 paired_g_contrast.py"""
import json, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
rows = json.load(open(f"{CACHE}/mbpp_rows.json"))
s = np.load(f"{CACHE}/mbpp_rm_s.npy")
code = [r[1] for r in rows]
y = np.array([int(r[2]) for r in rows]); qid = np.array([int(r[3]) for r in rows])
cell = np.array([r[4] for r in rows])

def format_features(c):
    lines = c.split("\n"); nonblank = [l for l in lines if l.strip()]
    comments = [l for l in lines if l.strip().startswith("#")]
    ncomm = sum(len(l) for l in comments)
    return [len(c), len(lines), len(c.split()), len(comments),
            ncomm / max(1, len(c)), 1.0 if '"""' in c else 0.0,
            len(lines) - len(nonblank),
            np.mean([len(l) for l in nonblank]) if nonblank else 0.0,
            max((len(l) for l in nonblank), default=0)]

phi = np.array([format_features(c) for c in code], float)
g = cross_val_predict(Ridge(alpha=1.0), phi, s, cv=GroupKFold(5), groups=qid)
srep = s - g

idx = {}
for i, (p, c) in enumerate(zip(qid, cell)):
    idx.setdefault(int(p), {})[c] = i

for fmt in ("terse", "verbose"):
    pairs = [(d[f"correct-{fmt}"], d[f"buggy-{fmt}"]) for d in idx.values()
             if f"correct-{fmt}" in d and f"buggy-{fmt}" in d]
    dphi = np.array([np.abs(phi[a] - phi[b]).max() for a, b in pairs])
    dg = np.array([g[a] - g[b] for a, b in pairs])
    draw = np.array([s[a] - s[b] for a, b in pairs])
    drep = np.array([srep[a] - srep[b] for a, b in pairs])
    flips = int(((draw > 0) != (drep > 0)).sum())
    print(f"\n[{fmt}] n={len(pairs)} pairs")
    print(f"  identical phi within pair: {(dphi == 0).sum()}/{len(pairs)}")
    print(f"  paired g-contrast  mean {dg.mean():+.4f}  sd {dg.std():.4f}  "
          f"max|.| {np.abs(dg).max():.4f}")
    print(f"  construct margin   raw {draw.mean():+.4f} -> repaired {drep.mean():+.4f}")
    print(f"  within-pair preference: raw {(draw>0).mean():.3f} -> "
          f"repaired {(drep>0).mean():.3f}; sign flips {flips} "
          f"({flips/len(pairs)*100:.1f}%)")
    same = dphi == 0
    if same.any():
        print(f"  among identical-phi pairs, sign flips: "
              f"{int(((draw[same]>0)!=(drep[same]>0)).sum())}")
