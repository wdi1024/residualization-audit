#!/usr/bin/env python3
"""Paired change intervals for the designed code audit (v170 review, M2).

The recorded Table 1 reports raw and residualized paired effects with their own
problem-level intervals. A residual interval covering zero does not by itself
show that the effect changed, so this script computes, per problem, the change
(residualized minus raw) in each paired effect and bootstraps problems:
  - four paired effects (format|correct, format|buggy, construct|terse, construct|verbose)
  - the absolute format asymmetry |format|correct| vs |format|buggy| and its change
  - margin changes for the two construct contrasts (non-inferiority reading)
Recorded point values are reproduced first as a referee check. Also run on
HumanEval and the large-v2 checkpoint where caches exist.
"""
import json, os, numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,'..'); CACHE=os.path.join(ROOT,'cache')
CV=KFold(5,shuffle=True,random_state=0); B=2000
def feats(c):
    lines=c.split("\n"); nonblank=[l for l in lines if l.strip()]; comments=[l for l in lines if l.strip().startswith("#")]
    return [len(c),len(lines),len(c.split()),len(comments),sum(len(l) for l in comments)/max(1,len(c)),1.0 if '"""' in c else 0.0,len(lines)-len(nonblank),np.mean([len(l) for l in nonblank]) if nonblank else 0.0,max((len(l) for l in nonblank),default=0)]
def run(rows_file,s_file,tag,ref=None):
    rows=json.load(open(os.path.join(CACHE,rows_file))); s=np.load(os.path.join(CACHE,s_file)).astype(float)
    code=[r[1] for r in rows]; y=np.array([int(r[2]) for r in rows]); qid=np.array([int(r[3]) for r in rows]); cell=np.array([r[4] for r in rows])
    phi=np.array([feats(c) for c in code],float); srep=s-cross_val_predict(Ridge(alpha=1.0),phi,s,cv=CV)
    verbose=np.isin(cell,["correct-verbose","buggy-verbose"]); correct=y==1
    masks={"format_given_correct":(verbose&correct,~verbose&correct),"format_given_buggy":(verbose&~correct,~verbose&~correct),"construct_given_terse":(correct&~verbose,~correct&~verbose),"construct_given_verbose":(correct&verbose,~correct&verbose)}
    qs=np.unique(qid); per={}
    for k,(hi,lo) in masks.items():
        raw=[];res=[]
        for t in qs:
            a=(qid==t)&hi; b=(qid==t)&lo
            if a.any() and b.any(): raw.append(s[a].mean()-s[b].mean()); res.append(srep[a].mean()-srep[b].mean())
        per[k]=(np.array(raw),np.array(res))
    rng=np.random.default_rng(0); out={"tag":tag,"n":len(rows),"n_problems":int(len(qs))}
    def ci(v): return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    for k,(raw,res) in per.items():
        m=len(raw); d=res-raw; bm=[]; braw=[]; bres=[]
        for _ in range(B):
            i=rng.integers(0,m,m); bm.append(d[i].mean()); braw.append(raw[i].mean()); bres.append(res[i].mean())
        out[k]={"raw":float(raw.mean()),"raw_ci":ci(braw),"res":float(res.mean()),"res_ci":ci(bres),"change":float(d.mean()),"change_ci":ci(bm),"n_problems":m}
    # absolute format asymmetry: | mean(format|correct) | - | mean(format|buggy) |  (problem-bootstrapped jointly on problems with both)
    fc_raw,fc_res=per["format_given_correct"]; fb_raw,fb_res=per["format_given_buggy"]
    m=min(len(fc_raw),len(fb_raw)); asym_raw=[];asym_res=[];asym_ch=[]
    for _ in range(B):
        i=rng.integers(0,m,m); ar=abs(fc_raw[i].mean()-fb_raw[i].mean()); rs=abs(fc_res[i].mean()-fb_res[i].mean()); asym_raw.append(ar); asym_res.append(rs); asym_ch.append(rs-ar)
    out["format_asymmetry"]={"raw":float(abs(fc_raw.mean()-fb_raw.mean())),"raw_ci":ci(asym_raw),"res":float(abs(fc_res.mean()-fb_res.mean())),"res_ci":ci(asym_res),"change":float(abs(fc_res.mean()-fb_res.mean())-abs(fc_raw.mean()-fb_raw.mean())),"change_ci":ci(asym_ch)}
    print(f"\n== {tag}: n={len(rows)} problems={len(qs)}")
    for k in masks: o=out[k]; print(f"  {k:24s} raw {o['raw']:+.3f} {o['raw_ci']}  res {o['res']:+.3f} {o['res_ci']}  change {o['change']:+.3f} {o['change_ci']}")
    o=out["format_asymmetry"]; print(f"  format asymmetry |c|-|b|: raw {o['raw']:.3f} {o['raw_ci']} res {o['res']:.3f} {o['res_ci']} change {o['change']:+.3f} {o['change_ci']}")
    if ref:
        for k,v in ref.items(): assert abs(out[k]['raw']-v[0])<3e-3 and abs(out[k]['res']-v[1])<3e-3, (k,out[k]['raw'],out[k]['res'],v)
        print("  referee: recorded raw/residual points reproduced")
    return out
res={}
res["mbpp_base"]=run("mbpp_rows.json","mbpp_rm_s.npy","MBPP, deberta-v3-base RM",ref={"format_given_correct":(0.083,-0.041),"format_given_buggy":(0.154,0.035),"construct_given_terse":(0.158,0.160)})
for rf,sf,tag in [("mbpp_rows.json","mbpp_rm_large_s.npy","MBPP, deberta-v3-large-v2 RM"),("humaneval_rows.json","humaneval_rm_s.npy","HumanEval, deberta-v3-base RM")]:
    if os.path.exists(os.path.join(CACHE,sf)): res[tag]=run(rf,sf,tag)
json.dump(res,open(os.path.join(HERE,"rm_code_paired_change.json"),"w"),indent=1); print("\nwrote grouped_rerun/rm_code_paired_change.json")

# ---- mirrored win-rate asymmetry (abstract: 0.507 vs 0.615 -> 0.586 vs 0.550) and a worked example ----
def mirrored(rows_file,s_file,tag):
    rows=json.load(open(os.path.join(CACHE,rows_file))); s=np.load(os.path.join(CACHE,s_file)).astype(float)
    code=[r[1] for r in rows]; y=np.array([int(r[2]) for r in rows]); qid=np.array([int(r[3]) for r in rows]); cell=np.array([r[4] for r in rows])
    phi=np.array([feats(c) for c in code],float); srep=s-cross_val_predict(Ridge(alpha=1.0),phi,s,cv=CV)
    qs=np.unique(qid); w1=[];w2=[];r1=[];r2=[]
    for t in qs:
        m=qid==t
        def pick(c): 
            i=np.where(m&(cell==c))[0]; return i[0] if len(i) else None
        ct,cv_,bt,bv=pick("correct-terse"),pick("correct-verbose"),pick("buggy-terse"),pick("buggy-verbose")
        if None in (ct,cv_,bt,bv): continue
        w1.append(float(s[ct]>s[bv])); w2.append(float(s[cv_]>s[bt])); r1.append(float(srep[ct]>srep[bv])); r2.append(float(srep[cv_]>srep[bt]))
    w1,w2,r1,r2=map(np.array,(w1,w2,r1,r2)); m=len(w1); rng=np.random.default_rng(1)
    raw=abs(w2.mean()-w1.mean()); res=abs(r2.mean()-r1.mean()); b_raw=[];b_res=[];b_ch=[]
    for _ in range(B):
        i=rng.integers(0,m,m); a=abs(w2[i].mean()-w1[i].mean()); c=abs(r2[i].mean()-r1[i].mean()); b_raw.append(a); b_res.append(c); b_ch.append(c-a)
    out={"n_problems":m,"win_terse_correct_vs_verbose_buggy":[float(w1.mean()),float(r1.mean())],"win_verbose_correct_vs_terse_buggy":[float(w2.mean()),float(r2.mean())],
         "asymmetry_raw":float(raw),"asymmetry_raw_ci":[float(np.percentile(b_raw,2.5)),float(np.percentile(b_raw,97.5))],"asymmetry_res":float(res),"asymmetry_res_ci":[float(np.percentile(b_res,2.5)),float(np.percentile(b_res,97.5))],"asymmetry_change":float(res-raw),"asymmetry_change_ci":[float(np.percentile(b_ch,2.5)),float(np.percentile(b_ch,97.5))]}
    print(f"\n== mirrored win rates ({tag}): wins {out['win_terse_correct_vs_verbose_buggy']} / {out['win_verbose_correct_vs_terse_buggy']} | asym raw {raw:.3f} {out['asymmetry_raw_ci']} res {res:.3f} {out['asymmetry_res_ci']} change {res-raw:+.3f} {out['asymmetry_change_ci']}")
    # worked example: a problem where raw prefers buggy-verbose over correct-terse but residual reverses, with moderate code length
    best=None
    for t in qs:
        m_=qid==t; idx={c:np.where(m_&(cell==c))[0] for c in ["correct-terse","correct-verbose","buggy-terse","buggy-verbose"]}
        if any(len(v)==0 for v in idx.values()): continue
        i={k:v[0] for k,v in idx.items()}
        if s[i["buggy-verbose"]]>s[i["correct-terse"]] and srep[i["correct-terse"]]>srep[i["buggy-verbose"]] and len(code[i["correct-terse"]])<260:
            cand={"qid":int(t),"prompt":rows[i["correct-terse"]][0][:200],"cells":{k:{"test":int(y[j]),"raw":float(s[j]),"res":float(srep[j]),"chars":len(code[j]),"code":code[j][:400]} for k,j in i.items()}}
            if best is None or len(code[i["correct-terse"]])<len(best["cells"]["correct-terse"]["code"]): best=cand
    out["example"]=best; return out
res=json.load(open(os.path.join(HERE,"rm_code_paired_change.json")))
res["mirrored_mbpp_base"]=mirrored("mbpp_rows.json","mbpp_rm_s.npy","MBPP base")
json.dump(res,open(os.path.join(HERE,"rm_code_paired_change.json"),"w"),indent=1)
ex=res["mirrored_mbpp_base"]["example"]; print("\nEXAMPLE qid",ex["qid"],"\nprompt:",ex["prompt"][:160])
for k,v in ex["cells"].items(): print(f"  {k:16s} test={v['test']} raw={v['raw']:+.3f} res={v['res']:+.3f} chars={v['chars']}\n     {v['code'][:220]!r}")
