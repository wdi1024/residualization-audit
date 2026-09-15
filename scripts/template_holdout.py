"""Reviewer round-6: does the format correction generalize to an unseen edit template?

The audit's verbose variants all come from one comment/docstring template. This
script builds a second, structurally different template (leading module comment,
Args/Returns docstring, a comment before the return, no trailing banner),
re-verifies every label by unit-test execution, scores the new candidates with the
same reward model, and asks whether a residualizer FIT ON THE ORIGINAL TEMPLATE
removes the new template's format effect. Nothing in the operator is refit on the
held-out template. Run: python3 template_holdout.py"""
import json, os, re, subprocess, sys, tempfile
import numpy as np
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "notes", "template_holdout.json")
TIMEOUT = 10

rows = json.load(open(f"{CACHE}/mbpp_rows.json"))
s_orig = np.load(f"{CACHE}/mbpp_rm_s.npy")
by = {}
for i, (q, code, y, pid, cell) in enumerate(rows):
    by.setdefault(int(pid), {})[cell] = i

def body_indent(lines, i):
    for ln in lines[i + 1:]:
        if ln.strip():
            return ln[:len(ln) - len(ln.lstrip())]
    return "    "

def verbosify_B(code):
    """Held-out template: comment/docstring-only, structurally unlike template A."""
    lines = code.replace("\r\n", "\n").split("\n")
    out = ["# Utility routine for the task stated above."]
    added_doc = False
    ret_i = max((k for k, ln in enumerate(lines) if ln.strip().startswith("return")),
                default=-1)
    for i, ln in enumerate(lines):
        st = ln.strip()
        if i == ret_i and st.startswith("return"):
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{ind}# hand back the value requested by the caller")
        out.append(ln)
        if not added_doc and st.startswith("def ") and st.endswith(":"):
            ind = body_indent(lines, i)
            out.append(f'{ind}"""Return the value requested by the task.')
            out.append(f"{ind}")
            out.append(f"{ind}Args:")
            out.append(f"{ind}    inputs as described in the problem statement.")
            out.append(f"{ind}Returns:")
            out.append(f"{ind}    the computed result.")
            out.append(f'{ind}"""')
            added_doc = True
    return "\n".join(out)

def run_tests(code, setup, tests):
    src = code + "\n" + (setup or "") + "\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True,
                              timeout=TIMEOUT).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)

from datasets import load_dataset
mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
# the audit's pid is a sequential index, not the MBPP task_id: match on the statement
tests_by_text = {r["text"]: (r.get("test_setup_code", ""), r["test_list"]) for r in mbpp}
print(f"MBPP tests loaded for {len(tests_by_text)} problems", flush=True)

new_codes, keys, dropped = [], [], 0
for pid, d in sorted(by.items()):
    if "correct-terse" not in d or "buggy-terse" not in d:
        dropped += 1; continue
    stmt = rows[d["correct-terse"]][0]
    if stmt not in tests_by_text:
        dropped += 1; continue
    setup, tests = tests_by_text[stmt]
    ok = True
    variants = {}
    for lab, cell in (("correct", "correct-terse"), ("buggy", "buggy-terse")):
        v = verbosify_B(rows[d[cell]][1])
        passes = run_tests(v, setup, tests)
        if passes != (lab == "correct"):     # label must survive the edit
            ok = False; break
        variants[lab] = v
    if not ok:
        dropped += 1; continue
    for lab in ("correct", "buggy"):
        new_codes.append((rows[d[f"{lab}-terse"]][0], variants[lab]))
        keys.append((pid, lab))
print(f"template-B candidates: {len(new_codes)} ({dropped} problems dropped on re-verification)", flush=True)

SC = os.path.join(CACHE, "mbpp_rm_templateB_s.npy")
if os.path.exists(SC):
    s_new = np.load(SC)
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "OpenAssistant/reward-model-deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    torch.set_num_threads(6)
    s_new = np.zeros(len(new_codes))
    with torch.no_grad():
        for b in range(0, len(new_codes), 32):
            chunk = new_codes[b:b+32]
            enc = tok([q for q, _ in chunk], [c for _, c in chunk], truncation=True,
                      padding=True, max_length=512, return_tensors="pt")
            s_new[b:b+32] = mdl(**enc).logits[:, 0].numpy()
            if b % 320 == 0: print(f"  scored {b}/{len(new_codes)}", flush=True)
    np.save(SC, s_new)

def fmt_feats(c):
    L = c.split("\n"); nb = [l for l in L if l.strip()]
    cm = [l for l in L if l.strip().startswith("#")]
    return [len(c), len(L), len(c.split()), len(cm),
            sum(len(l) for l in cm) / max(1, len(c)), 1.0 if '"""' in c else 0.0,
            len(L) - len(nb), np.mean([len(l) for l in nb]) if nb else 0.0,
            max((len(l) for l in nb), default=0)]

phi_A = np.array([fmt_feats(r[1]) for r in rows], float)
qid_A = np.array([int(r[3]) for r in rows])
phi_B = np.array([fmt_feats(c) for _, c in new_codes], float)

# operator fit on the ORIGINAL template only; applied unchanged to template B
g_A = cross_val_predict(Ridge(alpha=1.0), phi_A, s_orig, cv=GroupKFold(5), groups=qid_A)
g_model = Ridge(alpha=1.0).fit(phi_A, s_orig)
g_B = g_model.predict(phi_B)
srep_A, srep_B = s_orig - g_A, s_new - g_B

newidx = {k: i for i, k in enumerate(keys)}
pids = sorted({p for p, _ in keys})
def paired(vals_new, vals_old, lab):
    d = np.array([vals_new[newidx[(p, lab)]] - vals_old[by[p][f"{lab}-terse"]]
                  for p in pids])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(1000)]
    return float(d.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

res = {"n_problems": len(pids), "dropped": dropped, "template": "B (held out)"}
print(f"\nheld-out template, n={len(pids)} problems")
for lab in ("correct", "buggy"):
    raw = paired(s_new, s_orig, lab)
    rep = paired(srep_B, srep_A, lab)
    res[f"format_{lab}_raw"], res[f"format_{lab}_rep"] = raw, rep
    print(f"  format effect @ {lab:7s} raw {raw[0]:+.3f} [{raw[1][0]:+.3f},{raw[1][1]:+.3f}]"
          f"  ->  residualized {rep[0]:+.3f} [{rep[1][0]:+.3f},{rep[1][1]:+.3f}]")
# construct margins under the held-out template
def margin(vals, suffix):
    d = np.array([vals[newidx[(p, "correct")]] - vals[newidx[(p, "buggy")]] for p in pids])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(1000)]
    return float(d.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
mr, mrep = margin(s_new, "raw"), margin(srep_B, "rep")
res["construct_raw"], res["construct_rep"] = mr, mrep
print(f"  construct margin (new template) raw {mr[0]:+.3f} [{mr[1][0]:+.3f},{mr[1][1]:+.3f}]"
      f"  ->  residualized {mrep[0]:+.3f} [{mrep[1][0]:+.3f},{mrep[1][1]:+.3f}]")
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
