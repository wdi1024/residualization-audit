"""Screen false-negative (recall) experiment: run the FULL frozen pipeline on the
C1-screen-REJECTED candidates (rounds 1+2) to measure the screen's false-negative
rate — reviewers noted rejected candidates were never scored, so a candidate the
screen refused could in principle still be a validated positive (screen FN).

Candidates (screen C1 from notes/non_nli_screen_2026-07-10.md and
notes/second_positive_hunt_2026-07-09.md; all < 0.65 gate):
  shp        SHP preference          0.647 (borderline)  RM margin (OA deberta-v3-base)
  hatecheck  HateCheck               0.637               toxic-bert P(toxic)
  boolq      BoolQ                   0.640               qnli-distilroberta CE logit
  mnli       MNLI hyp-only           0.616               textattack SNLI-bert P(entail)
  rte        RTE hyp-only            0.524               nli-deberta-v3-small P(entail)
  paws       PAWS pair               0.538               textattack QQP-bert P(dup)
  hh         hh-rlhf                 0.513               RM margin (OA deberta-v3-base)

Frozen recipe, verbatim from the paper's runs (heldout_squad_confirmatory /
grouped_full): phi = the screen's exact feature recipe per candidate; artifact
model = cross-fitted LogisticRegression (StandardScaler for dense phi, plain for
TF-IDF, exactly as the screen); repair = cross-fitted Ridge(alpha=1) residual;
CV = GroupKFold(5) where a query/post/premise grouping exists (duplicate-group
fraction >= 0.05, deterministic first-occurrence group ids — NEVER hash()),
else KFold(5, shuffle, seed 0) as the screen used; slice threshold
rate-calibrated (quantile at 1 - base rate); gates C1 >= 0.65, C2 R2 >= 0.05,
Delta_adv >= 0.03, decorrelation |corr(s_rep, index)| <= 0.05; bootstrap CIs
1000 resamples (group-clustered where grouped).

Scorer pairings are construct-valid (off-distribution / error-capable; avoids
the QQP in-distribution trap and the RewardBench degenerate-label trap of
notes/DATA_ACQUISITION.md): rationale recorded per candidate in the committed
artifact and the summary note. SHP / hh-rlhf use OpenAssistant
reward-model-deberta-v3-base (training mix webgpt/summarize/synthetic-instruct
— excludes both SHP and hh-rlhf; large-v2's mix includes hh-rlhf, which would
be in-distribution) with the screen's exact pair construction (SHP: dataset
labels; hh: deterministic i%%2 flip), subsampled to n=2000 for CPU/MPS runtime
(allowed by protocol; n recorded).

Single-shot protocol: the committed artifact (candidate, scorer, features,
slice rule, gates, screen-based prediction "no validated positive expected")
is written BEFORE the scorer produces any score.

Determinism protocol: run once (writes notes/screen_recall_<cand>.json), then
run again with --confirm in a SEPARATE process: recomputes every statistic
(scores from the on-disk cache), re-scores a fixed 32-item spot-check batch,
and requires exact equality of all recorded values before stamping
"determinism_check" into the JSON.

Usage: python3 screen_recall_run.py <candidate> [--confirm]
Writes ../notes/screen_recall_<candidate>.json (+ .committed).
"""
import json
import os
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "..", "notes")
CACHE = os.path.join(HERE, "..", "cache")
os.makedirs(CACHE, exist_ok=True)
SCREEN_CV = KFold(5, shuffle=True, random_state=0)
GATES = {"C1": 0.65, "C2_R2": 0.05, "C2_Dadv": 0.03, "cert": 0.05}
IDENT = ["black", "white", "muslim", "jewish", "christian", "gay", "lesbian",
         "homosexual", "trans", "transgender", "women", "woman", "female",
         "men", "man", "male", "immigrant", "mexican", "islam", "muslims",
         "jews", "gays", "queer", "asian", "african", "catholic", "atheist",
         "feminist", "disabled", "disability"]


def lens(A, B):
    la = np.array([len(x.split()) for x in A], float)[:, None]
    lb = np.array([len(x.split()) for x in B], float)[:, None]
    return np.hstack([la, lb, la - lb, np.log1p(la) - np.log1p(lb)])


def first_occurrence_ids(keys):
    """Deterministic group ids (prior bug: hash() drifted across processes)."""
    seen = {}
    return np.array([seen.setdefault(k, len(seen)) for k in keys])


# ---------------- candidate builders ----------------
# each returns dict(pairs=[(text_a, text_b)...], y, phi, dense_or_none,
#                   groups, scale_c1, scorer, target, max_len, note)

def build_shp():
    from datasets import load_dataset
    d = load_dataset("stanfordnlp/SHP", split="test")
    rows = [(r["history"], r["human_ref_A"], r["human_ref_B"], r["labels"],
             r["post_id"]) for r in d][:4000]
    F4000 = lens([a for _, a, _, _, _ in rows], [b for _, _, b, _, _ in rows])
    y4000 = np.array([c for _, _, _, c, _ in rows])
    idx = np.sort(np.random.default_rng(0).choice(4000, 2000, replace=False))
    rows = [rows[i] for i in idx]
    y = np.array([c for _, _, _, c, _ in rows])
    F = lens([a for _, a, _, _, _ in rows], [b for _, _, b, _, _ in rows])
    return dict(pairs=[(h, a) for h, a, _, _, _ in rows],
                pairs_b=[(h, b) for h, _, b, _, _ in rows],
                y=y, phi=csr_matrix(F), dense=F,
                groups=first_occurrence_ids([p for *_, p in rows]),
                scale_c1=True, scorer="OpenAssistant/reward-model-deberta-v3-base",
                target="rm_margin", max_len=512,
                screen_c1_full=(F4000, y4000),
                note="n=2000 subsample of the screen's first-4000 (seed-0 choice); "
                     "phi = length-only [la,lb,la-lb,dlog1p]; post_id-grouped")


def build_hh():
    from datasets import load_dataset
    d = load_dataset("Anthropic/hh-rlhf", split="test")
    rows4000 = [(r["chosen"], r["rejected"]) for r in d][:4000]
    A, B, y = [], [], []
    for i, (c, rj) in enumerate(rows4000):        # screen's deterministic flip
        if i % 2 == 0:
            A.append(c); B.append(rj); y.append(1)
        else:
            A.append(rj); B.append(c); y.append(0)
    y = np.array(y)
    F4000 = lens(A, B)
    idx = np.sort(np.random.default_rng(0).choice(4000, 2000, replace=False))
    A = [A[i] for i in idx]; B = [B[i] for i in idx]; y = y[idx]
    F = lens(A, B)

    def split_hh(t):
        k = t.rfind("\n\nAssistant:")
        return (t[:k].strip(), t[k + len("\n\nAssistant:"):].strip()) if k >= 0 \
            else ("", t.strip())
    return dict(pairs=[split_hh(a) for a in A], pairs_b=[split_hh(b) for b in B],
                y=y, phi=csr_matrix(F), dense=F,
                groups=first_occurrence_ids([split_hh(a)[0] for a in A]),
                scale_c1=True, scorer="OpenAssistant/reward-model-deberta-v3-base",
                target="rm_margin", max_len=512,
                screen_c1_full=(F4000, np.array([1, 0] * 2000)),
                note="n=2000 subsample (seed-0 choice) of screen's flip-constructed "
                     "4000; phi = length-only on FULL dialogue strings (screen "
                     "verbatim); s = RM margin on last-assistant-turn split")


def build_hatecheck():
    from datasets import load_dataset
    d = load_dataset("Paul/hatecheck", split="test")
    texts = [r["test_case"] for r in d]
    y = np.array([1 if r["label_gold"] == "hateful" else 0 for r in d])

    def ic(t):
        tl = t.lower()
        return [len(re.findall(r"\b" + re.escape(w) + r"\b", tl)) for w in IDENT]
    F = np.hstack([np.array([ic(t) for t in texts], float),
                   np.array([len(t.split()) for t in texts], float)[:, None]])
    return dict(pairs=[(t, None) for t in texts], y=y, phi=csr_matrix(F),
                dense=F, groups=None, scale_c1=True,
                scorer="unitary/toxic-bert", target="sigmoid:toxic", max_len=128,
                note="full n=3728; phi = identity-term counts (30) + length "
                     "(screen verbatim); no query grouping (templated cases)")


def build_boolq():
    from datasets import load_dataset
    d = load_dataset("google/boolq", split="train")
    rows = [(r["question"], r["passage"], 1 if r["answer"] else 0)
            for r in d][:6000]
    q = [a for a, _, _ in rows]
    y = np.array([c for _, _, c in rows])
    Xq = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(q)
    return dict(pairs=[(a, b) for a, b, _ in rows], y=y, phi=Xq, dense=None,
                groups=None, scale_c1=False,
                scorer="cross-encoder/qnli-distilroberta-base",
                target="logit", max_len=512,
                note="full n=6000; phi = question-only TF-IDF 8000 1-2gram "
                     "(screen verbatim); questions unique -> no grouping")


def build_mnli():
    from datasets import load_dataset
    d = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    rows = [(r["premise"], r["hypothesis"], 1 if r["label"] == 0 else 0)
            for r in d if r["label"] in (0, 2)][:4000]
    hyp = [b for _, b, _ in rows]
    y = np.array([c for _, _, c in rows])
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
    return dict(pairs=[(a, b) for a, b, _ in rows], y=y, phi=Xh, dense=None,
                groups=first_occurrence_ids([a for a, _, _ in rows]),
                scale_c1=False, scorer="textattack/bert-base-uncased-snli",
                target="softmax:1", max_len=128,
                note="full n=4000 entail-vs-contra; phi = hyp-only TF-IDF "
                     "(screen verbatim); premise-grouped; entail = class idx 1 "
                     "(second_positive_hunt sec.4)")


def build_rte():
    from datasets import load_dataset
    d = load_dataset("nyu-mll/glue", "rte", split="train")
    rows = [(r["sentence1"], r["sentence2"], 1 - r["label"]) for r in d]
    y = np.array([c for _, _, c in rows])
    Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(
        [b for _, b, _ in rows])
    return dict(pairs=[(a, b) for a, b, _ in rows], y=y, phi=Xh, dense=None,
                groups=first_occurrence_ids([a for a, _, _ in rows]),
                scale_c1=False, scorer="cross-encoder/nli-deberta-v3-small",
                target="softmax:entailment", max_len=256,
                note="full n=2490; phi = sentence2(hyp)-only TF-IDF "
                     "(screen verbatim)")


def build_paws():
    from datasets import load_dataset
    d = load_dataset("google-research-datasets/paws", "labeled_final",
                     split="train")
    rows = [(r["sentence1"], r["sentence2"], r["label"]) for r in d][:4000]
    y = np.array([c for _, _, c in rows])
    Xp = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(
        [a + " [SEP] " + b for a, b, _ in rows])
    return dict(pairs=[(a, b) for a, b, _ in rows], y=y, phi=Xp, dense=None,
                groups=first_occurrence_ids([a for a, _, _ in rows]),
                scale_c1=False, scorer="textattack/bert-base-uncased-QQP",
                target="softmax:1", max_len=128,
                note="full n=4000; phi = pair-text TF-IDF (screen verbatim); "
                     "sentence1-grouped; QQP-bert on PAWS = the canonical "
                     "artifact-corrupted off-distribution pairing")


CANDS = {"shp": (build_shp, 0.647), "hatecheck": (build_hatecheck, 0.637),
         "boolq": (build_boolq, 0.640), "mnli": (build_mnli, 0.616),
         "rte": (build_rte, 0.524), "paws": (build_paws, 0.538),
         "hh": (build_hh, 0.513)}

RATIONALE = {
    "shp": "OA deberta-v3-base RM trained on webgpt/summarize/synthetic-instruct"
           " -> SHP (Reddit) is off-distribution; RM length bias is the"
           " documented artifact; y = SHP human preference (independent).",
    "hh": "base RM (NOT large-v2, whose training mix includes hh-rlhf ->"
          " in-distribution trap); hh-rlhf off-distribution for base; screen's"
          " deterministic flip keeps y non-degenerate (RewardBench trap avoided).",
    "hatecheck": "toxic-bert (Jigsaw-trained) on HateCheck functional tests that"
                 " are DESIGNED to trip identity-term-reliant classifiers ->"
                 " off-distribution, artifact-driven error possible.",
    "boolq": "qnli-distilroberta cross-encoder (QNLI answerability) applied to"
             " BoolQ yes/no -> off-distribution scorer whose score can lean on"
             " surface question cues; y = BoolQ gold answer (independent).",
    "mnli": "SNLI-trained bert applied to MNLI (off-distribution, mirrors the"
            " SNLI positive's model-perpendicular-data recipe).",
    "rte": "SNLI+MNLI-trained nli-deberta-v3-small applied to RTE"
           " (off-distribution, substantial error rate).",
    "paws": "QQP-trained bert applied to PAWS, which was constructed so that"
            " overlap-reliant QQP models fail -> score genuinely"
            " artifact-corrupted (C2 expected to hold; C1 is what failed).",
}


# ---------------- scoring ----------------
def get_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def score_batched(model_name, pairs, target, max_len, batch=16, limit=None):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(6)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(dev)
    if target.startswith("softmax:"):
        key = target.split(":", 1)[1]
        if key.isdigit():
            ti = int(key)
        else:
            ti = next(i for i, l in model.config.id2label.items()
                      if key in l.lower())
    elif target.startswith("sigmoid:"):
        key = target.split(":", 1)[1]
        ti = next(i for i, l in model.config.id2label.items()
                  if l.lower() == key)
    pairs = pairs[:limit] if limit else pairs
    out = np.zeros(len(pairs))
    with torch.no_grad():
        for b in range(0, len(pairs), batch):
            chunk = pairs[b:b + batch]
            if chunk[0][1] is None:
                enc = tok([a for a, _ in chunk], truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")
            else:
                enc = tok([a for a, _ in chunk], [bb for _, bb in chunk],
                          truncation=True, padding=True, max_length=max_len,
                          return_tensors="pt")
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.cpu()
            if target == "logit":
                out[b:b + len(chunk)] = logits[:, 0].numpy()
            elif target.startswith("softmax:"):
                out[b:b + len(chunk)] = logits.softmax(-1)[:, ti].numpy()
            elif target.startswith("sigmoid:"):
                out[b:b + len(chunk)] = logits.sigmoid()[:, ti].numpy()
            elif target == "rm_margin":
                out[b:b + len(chunk)] = logits[:, 0].numpy()
            if b % (batch * 32) == 0:
                print(f"  scored {b}/{len(pairs)}", flush=True)
    del model
    return out


def get_scores(cand, D, confirm):
    cache = os.path.join(CACHE, f"screen_recall_{cand}_s.npy")
    spot = {}
    if os.path.exists(cache):
        s = np.load(cache)
        assert len(s) == len(D["y"]), "cache length mismatch"
        if confirm:   # re-score fixed 32-item spot-check batch, same batching
            k = min(32, len(D["y"]))
            sa = score_batched(D["scorer"], D["pairs"], D["target"],
                               D["max_len"], limit=k)
            if D["target"] == "rm_margin":
                sb = score_batched(D["scorer"], D["pairs_b"], D["target"],
                                   D["max_len"], limit=k)
                sa = sa - sb
            spot = {"spot_check_n": k,
                    "spot_check_max_absdiff": float(np.abs(sa - s[:k]).max())}
            assert spot["spot_check_max_absdiff"] < 5e-4, \
                f"scorer nondeterminism: {spot}"
        return s, spot
    sa = score_batched(D["scorer"], D["pairs"], D["target"], D["max_len"])
    if D["target"] == "rm_margin":
        sb = score_batched(D["scorer"], D["pairs_b"], D["target"], D["max_len"])
        sa = sa - sb
    np.save(cache, sa)
    return sa, spot


# ---------------- frozen pipeline ----------------
def run(cand, confirm=False):
    t0 = time.time()
    build, screen_c1 = CANDS[cand]
    D = build()
    y, phi = D["y"], D["phi"]
    n = len(y)

    # C1 re-measurement, screen CV verbatim (ungrouped KFold, screen classifier)
    def c1_model():
        clf = LogisticRegression(max_iter=1000)
        return make_pipeline(StandardScaler(with_mean=False), clf) \
            if D["scale_c1"] else clf
    Xc1 = D["dense"] if D["dense"] is not None else phi
    p_screen = cross_val_predict(c1_model(), Xc1, y, cv=SCREEN_CV,
                                 method="predict_proba")[:, 1]
    c1 = float(roc_auc_score(y, p_screen))
    c1_full = None
    if "screen_c1_full" in D:   # re-measure at the screen's original n too
        Ff, yf = D["screen_c1_full"]
        pf = cross_val_predict(c1_model(), Ff, yf, cv=SCREEN_CV,
                               method="predict_proba")[:, 1]
        c1_full = float(roc_auc_score(yf, pf))

    # grouping rule: grouped iff duplicate-group fraction >= 0.05
    groups = D["groups"]
    grouped = groups is not None and (1 - len(np.unique(groups)) / n) >= 0.05
    cv_kw = dict(cv=GroupKFold(5), groups=groups) if grouped \
        else dict(cv=SCREEN_CV)

    # ---- committed artifact BEFORE scoring ----
    cfile = os.path.join(NOTES, f"screen_recall_{cand}.json.committed")
    committed = {
        "candidate": cand, "n": n, "scorer": D["scorer"],
        "score_def": D["target"], "phi": D["note"],
        "pairing_rationale": RATIONALE[cand],
        "slice_rule": "rate-calibrated: adv = (art_prob > quantile(art_prob, "
                      "1-mean(y))) != (y==1)",
        "cv": ("GroupKFold(5), deterministic first-occurrence group ids"
               if grouped else "KFold(5, shuffle, seed 0)"),
        "gates": GATES,
        "screen_C1": screen_c1, "remeasured_C1": c1,
        "remeasured_C1_at_screen_n": c1_full,
        "committed_prediction": "no validated positive expected "
                                f"(screen C1 {screen_c1:.3f} < 0.65 -> refusal)",
    }
    if os.path.exists(cfile):
        prev = json.load(open(cfile))
        assert prev["remeasured_C1"] == committed["remeasured_C1"], \
            f"C1 drift across processes: {prev['remeasured_C1']} vs {c1}"
    else:
        json.dump(committed, open(cfile, "w"), indent=1)
        print("COMMITTED BEFORE SCORING:", json.dumps(committed), flush=True)

    # ---- score ----
    s, spot = get_scores(cand, D, confirm)

    # ---- pipeline (grouped where applicable) ----
    art = cross_val_predict(c1_model(), Xc1, y, method="predict_proba",
                            **cv_kw)[:, 1]
    index = cross_val_predict(c1_model(), Xc1, y, method="decision_function",
                              **cv_kw)
    c1_pipe = float(roc_auc_score(y, art))
    thr = float(np.quantile(art, 1 - y.mean()))
    adv = (art > thr) != (y == 1)
    full = np.ones(n, bool)
    pred = cross_val_predict(Ridge(alpha=1.0), phi, s, **cv_kw)
    srep = s - pred
    r2 = 1 - float(((s - pred) ** 2).sum()) / float(((s - s.mean()) ** 2).sum())

    def A(m, u):
        return float(roc_auc_score(y[m], u[m])) if y[m].min() != y[m].max() \
            else None
    dadv = A(full, s) - A(adv, s)

    rng = np.random.default_rng(0)

    gidx = {g: np.where(groups == g)[0] for g in np.unique(groups)} \
        if grouped else None

    def gain_ci(mask, B=1000):
        gains = []
        if grouped:
            gs = np.array(sorted(gidx))
            for _ in range(B):
                take = rng.choice(gs, len(gs), replace=True)
                idx = np.concatenate([gidx[t] for t in take])
                m = mask[idx]
                ya, sa, ra = y[idx][m], s[idx][m], srep[idx][m]
                if len(ya) == 0 or ya.min() == ya.max():
                    continue
                gains.append(roc_auc_score(ya, ra) - roc_auc_score(ya, sa))
        else:
            ii = np.where(mask)[0]
            for _ in range(B):
                b = rng.choice(ii, len(ii), replace=True)
                if y[b].min() == y[b].max():
                    continue
                gains.append(roc_auc_score(y[b], srep[b])
                             - roc_auc_score(y[b], s[b]))
        return [float(np.percentile(gains, 2.5)),
                float(np.percentile(gains, 97.5))]

    def corr_ci(u, v, B=1000):
        vals = []
        for _ in range(B):
            ii = rng.integers(0, n, n)
            vals.append(abs(float(np.corrcoef(u[ii], v[ii])[0, 1])))
        return [float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    adv_ci = gain_ci(adv)
    full_ci = gain_ci(full)
    cert_before = abs(float(np.corrcoef(s, index)[0, 1]))
    cert_after = abs(float(np.corrcoef(srep, index)[0, 1]))
    cert_after_ci = corr_ci(srep, index)

    c2_pass = bool(r2 >= GATES["C2_R2"] and dadv >= GATES["C2_Dadv"])
    cert_pass = bool(cert_after <= GATES["cert"])
    gain = A(adv, srep) - A(adv, s)
    sig_pos = bool(adv_ci[0] > 0)
    validated = bool(c2_pass and cert_pass and sig_pos)
    res = {
        "candidate": cand, "n": n, "pos_rate": float(y.mean()),
        "grouped_cv": grouped, "n_groups": int(len(np.unique(groups)))
        if groups is not None else None,
        "screen_C1": screen_c1, "C1_remeasured_screen_cv": c1,
        "C1_remeasured_at_screen_n": c1_full, "C1_pipeline_cv": c1_pipe,
        "C1_gate_pass": bool(c1 >= GATES["C1"]),
        "s_raw_AUC_full": A(full, s),
        "C2_R2_phi_to_s": float(r2), "C2_Delta_adv": float(dadv),
        "C2_pass": c2_pass,
        "A_full": [A(full, s), A(full, srep)],
        "A_adv": [A(adv, s), A(adv, srep)], "n_adv": int(adv.sum()),
        "slice_gain": float(gain), "slice_gain_CI": adv_ci,
        "full_gain": float(A(full, srep) - A(full, s)), "full_gain_CI": full_ci,
        "decorrelation_before_after": [cert_before, cert_after],
        "cert_after_CI": cert_after_ci, "cert_pass": cert_pass,
        "pipeline_ran_to_completion": True,
        "verdict": ("VALIDATED POSITIVE (screen FN)" if validated else
                    "true negative: " + "; ".join(
                        ([] if c2_pass else [f"C2 fails (R2={r2:.3f}, "
                                             f"Dadv={dadv:.3f})"]) +
                        ([] if cert_pass else
                         [f"decorrelation fails ({cert_after:.3f}>0.05)"]) +
                        ([] if sig_pos else
                         [f"slice gain not significantly positive "
                          f"({gain:+.3f} CI {adv_ci})"]))),
        "screen_FN": validated,
        "committed": committed,
    }
    res.update(spot)
    out = os.path.join(NOTES, f"screen_recall_{cand}.json")
    if confirm:
        prev = json.load(open(out))
        prev_det = prev.pop("determinism_check", None)
        prev_rt = prev.pop("runtime_s", None)
        prev_spot = {k: prev.pop(k, None) for k in
                     ("spot_check_n", "spot_check_max_absdiff")}
        now = {k: v for k, v in res.items()
               if k not in ("spot_check_n", "spot_check_max_absdiff")}
        mismatch = [k for k in now if json.dumps(now[k], sort_keys=True)
                    != json.dumps(prev.get(k), sort_keys=True)]
        assert not mismatch, f"DETERMINISM FAILURE on {mismatch}"
        res["determinism_check"] = ("PASS: all statistics identical across two "
                                    "processes; scorer spot-check max|diff|="
                                    f"{res.get('spot_check_max_absdiff')}")
        res["runtime_s"] = prev_rt
        json.dump(res, open(out, "w"), indent=1)
        print(f"[{cand}] determinism CONFIRMED, stamped {out}", flush=True)
    else:
        res["runtime_s"] = round(time.time() - t0, 1)
        json.dump(res, open(out, "w"), indent=1)
        print(json.dumps(res, indent=1), flush=True)
        print("wrote", out, flush=True)
    return res


if __name__ == "__main__":
    cand = sys.argv[1]
    run(cand, confirm="--confirm" in sys.argv)
