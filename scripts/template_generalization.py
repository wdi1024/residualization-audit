"""Pre-registered (notes/prereg_template_generalization_20260908.md): does the format
correction generalize across edit templates?

Builds two further comment/docstring-only templates, C (inline comments, no docstring) and
D (header docstring, no inline comments), re-verifies every label by executing the problem's
MBPP unit tests, scores them with the same reward model, and runs three fits:

  (1) leave-one-template-out   fit on the other three, apply unrefit to the held-out one
  (2) fit on A only            the operator already reported in the paper, applied to C and D
  (3) fit on A+B               does a second training template close the gap?

Run: python3 template_generalization.py
"""
import json, os, re, subprocess, sys, tempfile
import numpy as np
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "notes", "template_generalization.json")
TIMEOUT = 10
SEED = 0

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
    """Template B, reproduced from template_holdout.py so all four share one pipeline."""
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


def _triple_quote_mask(lines):
    """True on lines that sit inside a triple-quoted literal (crude but sufficient)."""
    inside, mask, delim = False, [], None
    for ln in lines:
        mask.append(inside)
        i = 0
        while i < len(ln):
            if not inside and (ln.startswith('"""', i) or ln.startswith("'''", i)):
                inside, delim = True, ln[i:i + 3]
                mask[-1] = False
                i += 3
                continue
            if inside and ln.startswith(delim, i):
                inside = False
                i += 3
                continue
            i += 1
    return mask


_COMMENTS = ["# step in the computation",
             "# carry the running result forward",
             "# handle this part of the task",
             "# continue with the next stage",
             "# as required by the problem statement"]


def verbosify_C(code):
    """Template C: end-of-line comments on statements, no docstring anywhere."""
    lines = code.replace("\r\n", "\n").split("\n")
    inlit = _triple_quote_mask(lines)
    out, k = [], 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        skip = (inlit[i] or not st or st.startswith("#") or ln.rstrip().endswith("\\")
                or '"""' in ln or "'''" in ln or "#" in ln)
        if skip:
            out.append(ln)
            continue
        out.append(f"{ln}  {_COMMENTS[k % len(_COMMENTS)]}")
        k += 1
    return "\n".join(out)


def verbosify_D(code):
    """Template D: module-level docstring header and blank-line padding, no '#' comments."""
    lines = code.replace("\r\n", "\n").split("\n")
    inlit = _triple_quote_mask(lines)
    header = ['"""Reference implementation for the task stated above.',
              "",
              "The routine below follows the specification directly and returns the value",
              "the task asks for, without any additional reporting or side effects.",
              '"""',
              ""]
    out = list(header)
    for i, ln in enumerate(lines):
        out.append(ln)
        st = ln.strip()
        if (not inlit[i] and st.endswith(":") and not st.startswith(("def ", "class "))):
            out.append("")
    return "\n".join(out)


def run_tests(code, setup, tests):
    src = code + "\n" + (setup or "") + "\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True,
                              timeout=TIMEOUT).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


from datasets import load_dataset
mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
tests_by_text = {r["text"]: (r.get("test_setup_code", ""), r["test_list"]) for r in mbpp}
print(f"MBPP tests loaded for {len(tests_by_text)} problems", flush=True)

GENS = {"B": verbosify_B, "C": verbosify_C, "D": verbosify_D}


def build(tag, gen):
    """Return (codes, keys, dropped): label-verified variants for one template."""
    codes, keys, dropped = [], [], 0
    for pid, d in sorted(by.items()):
        if "correct-terse" not in d or "buggy-terse" not in d:
            dropped += 1
            continue
        stmt = rows[d["correct-terse"]][0]
        if stmt not in tests_by_text:
            dropped += 1
            continue
        setup, tests = tests_by_text[stmt]
        variants, ok = {}, True
        for lab, cell in (("correct", "correct-terse"), ("buggy", "buggy-terse")):
            v = gen(rows[d[cell]][1])
            if run_tests(v, setup, tests) != (lab == "correct"):
                ok = False
                break
            variants[lab] = v
        if not ok:
            dropped += 1
            continue
        for lab in ("correct", "buggy"):
            codes.append((rows[d[f"{lab}-terse"]][0], variants[lab]))
            keys.append((pid, lab))
    print(f"  template {tag}: {len(codes)} candidates, {dropped} problems dropped",
          flush=True)
    return codes, keys, dropped


def score(tag, codes):
    path = os.path.join(CACHE, f"mbpp_rm_template{tag}_s.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if len(arr) == len(codes):
            print(f"  template {tag}: cached scores ({len(arr)})", flush=True)
            return arr
        print(f"  template {tag}: cache size {len(arr)} != {len(codes)}, rescoring",
              flush=True)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "OpenAssistant/reward-model-deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    torch.set_num_threads(6)
    s = np.zeros(len(codes))
    with torch.no_grad():
        for b in range(0, len(codes), 32):
            chunk = codes[b:b + 32]
            enc = tok([q for q, _ in chunk], [c for _, c in chunk], truncation=True,
                      padding=True, max_length=512, return_tensors="pt")
            s[b:b + 32] = mdl(**enc).logits[:, 0].numpy()
            if b % 320 == 0:
                print(f"    scored {b}/{len(codes)}", flush=True)
    np.save(path, s)
    return s


def fmt_feats(c):
    L = c.split("\n")
    nb = [l for l in L if l.strip()]
    cm = [l for l in L if l.strip().startswith("#")]
    return [len(c), len(L), len(c.split()), len(cm),
            sum(len(l) for l in cm) / max(1, len(c)), 1.0 if '"""' in c else 0.0,
            len(L) - len(nb), np.mean([len(l) for l in nb]) if nb else 0.0,
            max((len(l) for l in nb), default=0)]


print("building templates", flush=True)
T = {}
for tag, gen in GENS.items():
    codes, keys, dropped = build(tag, gen)
    T[tag] = {"codes": codes, "keys": keys, "dropped": dropped,
              "s": score(tag, codes),
              "phi": np.array([fmt_feats(c) for _, c in codes], float),
              "idx": {k: i for i, k in enumerate(keys)}}

# template A comes from the original design; terse rows are the shared baseline
phi_all = np.array([fmt_feats(r[1]) for r in rows], float)
qid_all = np.array([int(r[3]) for r in rows])
terse_i = [i for i, r in enumerate(rows) if r[4].endswith("-terse")]
vA_i = [i for i, r in enumerate(rows) if r[4].endswith("-verbose")]
T["A"] = {"keys": [(int(rows[i][3]), rows[i][4].split("-")[0]) for i in vA_i],
          "s": s_orig[vA_i], "phi": phi_all[vA_i], "dropped": 0}
T["A"]["idx"] = {k: i for i, k in enumerate(T["A"]["keys"])}
print(f"  template A: {len(vA_i)} candidates (from the original design)", flush=True)

# feature-space check: the new templates must move phi in different directions
base = phi_all[terse_i].mean(0)
print("\nmean feature shift from terse (9 format features):", flush=True)
for tag in ("A", "B", "C", "D"):
    d = T[tag]["phi"].mean(0) - base
    print(f"  {tag}: " + " ".join(f"{v:+8.2f}" for v in d), flush=True)


def pool(train_tags):
    """Rows the residualizer is fit on: the shared terse rows plus the training templates."""
    phi = [phi_all[terse_i]]
    s = [s_orig[terse_i]]
    g = [qid_all[terse_i]]
    for tg in train_tags:
        phi.append(T[tg]["phi"])
        s.append(T[tg]["s"])
        g.append(np.array([p for p, _ in T[tg]["keys"]]))
    return np.vstack(phi), np.concatenate(s), np.concatenate(g)


def boot(d):
    rng = np.random.default_rng(SEED)
    b = [rng.choice(d, len(d), replace=True).mean() for _ in range(1000)]
    return float(d.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def evaluate(train_tags, held):
    """Format effect on `held`, raw and residualized, with the operator fit on train_tags."""
    phi_p, s_p, g_p = pool(train_tags)
    g_cv = cross_val_predict(Ridge(alpha=1.0), phi_p, s_p, cv=GroupKFold(5), groups=g_p)
    srep_terse = (s_p - g_cv)[:len(terse_i)]
    model = Ridge(alpha=1.0).fit(phi_p, s_p)
    H = T[held]
    srep_h = H["s"] - model.predict(H["phi"])
    terse_idx = {(int(rows[i][3]), rows[i][4].split("-")[0]): k
                 for k, i in enumerate(terse_i)}
    pids = sorted({p for p, _ in H["keys"]})
    res = {}
    for lab in ("correct", "buggy"):
        pp = [p for p in pids if (p, lab) in H["idx"] and (p, lab) in terse_idx]
        raw = np.array([H["s"][H["idx"][(p, lab)]] - s_orig[terse_i[terse_idx[(p, lab)]]]
                        for p in pp])
        rep = np.array([srep_h[H["idx"][(p, lab)]] - srep_terse[terse_idx[(p, lab)]]
                        for p in pp])
        res[f"format_{lab}_raw"] = boot(raw)
        res[f"format_{lab}_rep"] = boot(rep)
        res[f"format_{lab}_change"] = boot(rep - raw)
    pp = [p for p in pids if (p, "correct") in H["idx"] and (p, "buggy") in H["idx"]]
    res["construct_raw"] = boot(np.array(
        [H["s"][H["idx"][(p, "correct")]] - H["s"][H["idx"][(p, "buggy")]] for p in pp]))
    res["construct_rep"] = boot(np.array(
        [srep_h[H["idx"][(p, "correct")]] - srep_h[H["idx"][(p, "buggy")]] for p in pp]))
    res["n_problems"] = len(pp)
    return res


def show(name, r):
    print(f"\n{name}  (n={r['n_problems']})", flush=True)
    for lab in ("correct", "buggy"):
        a, b_ = r[f"format_{lab}_raw"], r[f"format_{lab}_rep"]
        c = r[f"format_{lab}_change"]
        print(f"  format @ {lab:7s} raw {a[0]:+.3f} [{a[1][0]:+.3f},{a[1][1]:+.3f}]"
              f"  ->  resid {b_[0]:+.3f} [{b_[1][0]:+.3f},{b_[1][1]:+.3f}]"
              f"   change {c[0]:+.3f} [{c[1][0]:+.3f},{c[1][1]:+.3f}]", flush=True)
    cr, cp = r["construct_raw"], r["construct_rep"]
    print(f"  construct margin raw {cr[0]:+.3f} -> resid {cp[0]:+.3f}"
          f" (change {cp[0]-cr[0]:+.3f})", flush=True)


results = {"prereg": "notes/prereg_template_generalization_20260908.md",
           "dropped": {t: T[t]["dropped"] for t in ("A", "B", "C", "D")},
           "analysis_1_leave_one_out": {}, "analysis_2_fit_A": {},
           "analysis_3_fit_AB": {}}

print("\n=== analysis 1: leave-one-template-out ===", flush=True)
for held in ("A", "B", "C", "D"):
    train = [t for t in ("A", "B", "C", "D") if t != held]
    r = evaluate(train, held)
    results["analysis_1_leave_one_out"][held] = r
    show(f"fit {'+'.join(train)} -> held-out {held}", r)

print("\n=== analysis 2: fit on A only (the operator reported in the paper) ===", flush=True)
for held in ("B", "C", "D"):
    r = evaluate(["A"], held)       # pool = terse rows + template A verbose
    results["analysis_2_fit_A"][held] = r
    show(f"fit A -> held-out {held}", r)

print("\n=== analysis 3: fit on A+B ===", flush=True)
for held in ("C", "D"):
    r = evaluate(["B"], held)
    results["analysis_3_fit_AB"][held] = r
    show(f"fit A+B -> held-out {held}", r)

json.dump(results, open(OUT, "w"), indent=1)
print("\nwrote", OUT, flush=True)
