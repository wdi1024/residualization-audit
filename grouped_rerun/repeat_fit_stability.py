#!/usr/bin/env python3
"""Full-procedure stability under repeated grouped fitting (v171 review, M10).
The recorded intervals hold the fitted pipeline fixed and resample evaluation rows. Here the
whole grouped pipeline (surface predictor, slice, ridge residualizer) is refit R times under
different group-to-fold assignments, and the slice gain G, full-population change and R^2
are recorded per refit. Also the code audit under R different KFold seeds."""
import json, os, sys, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pr_common import load, features, HERE, CACHE
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
R=20
def one(name,seed,q,ans,y,s,qid,dense,phi):
    rng=np.random.default_rng(seed); perm={g:i for i,g in enumerate(rng.permutation(np.unique(qid)))}; gid=np.array([perm[g] for g in qid])
    cv=GroupKFold(5); pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    art=cross_val_predict(pipe,dense,y,cv=cv,groups=gid,method="predict_proba")[:,1]
    thr=np.quantile(art,1-y.mean()); z=art>thr; adv=z!=(y==1)
    pred=cross_val_predict(Ridge(alpha=1.0),phi,s,cv=cv,groups=gid); srep=s-pred
    r2=1-float(((s-pred)**2).sum())/float(((s-s.mean())**2).sum())
    return {"G":float(roc_auc_score(y[adv],srep[adv])-roc_auc_score(y[adv],s[adv])),"full":float(roc_auc_score(y,srep)-roc_auc_score(y,s)),"R2":r2,"n_adv":int(adv.sum()),"C1":float(roc_auc_score(y,art))}
out={}
for name in ["squad","duorc","wikiqa","asnq","triviaqa","nq"]:
    try: q,ans,y,s,qid,_=load(name)
    except Exception as e: print(name,"skip",e,flush=True); continue
    s=np.asarray(s,float); dense,phi=features(q,ans); runs=[one(name,k,q,ans,y,s,qid,dense,phi) for k in range(R)]
    G=np.array([r["G"] for r in runs]); F=np.array([r["full"] for r in runs])
    out[name]={"R":R,"G_mean":float(G.mean()),"G_sd":float(G.std()),"G_min":float(G.min()),"G_max":float(G.max()),"G_all_positive":bool((G>0).all()),"full_mean":float(F.mean()),"full_sd":float(F.std()),"R2_mean":float(np.mean([r["R2"] for r in runs])),"runs":runs}
    print(name,{k:round(v,4) if isinstance(v,float) else v for k,v in out[name].items() if k!="runs"},flush=True)
# code audit: KFold seed repeats
import json as _j
rows=_j.load(open(os.path.join(CACHE,"mbpp_rows.json"))); sc=np.load(os.path.join(CACHE,"mbpp_rm_s.npy")).astype(float)
code=[r[1] for r in rows]; yy=np.array([int(r[2]) for r in rows]); pid=np.array([int(r[3]) for r in rows]); cell=np.array([r[4] for r in rows])
def feats(c):
    lines=c.split("\n"); nb=[l for l in lines if l.strip()]; cm=[l for l in lines if l.strip().startswith("#")]
    return [len(c),len(lines),len(c.split()),len(cm),sum(len(l) for l in cm)/max(1,len(c)),1.0 if '"""' in c else 0.0,len(lines)-len(nb),np.mean([len(l) for l in nb]) if nb else 0.0,max((len(l) for l in nb),default=0)]
ph=np.array([feats(c) for c in code],float); verbose=np.isin(cell,["correct-verbose","buggy-verbose"]); correct=yy==1
def paired(u,hi,lo):
    v=[]
    for t in np.unique(pid):
        a=u[(pid==t)&hi]; b=u[(pid==t)&lo]
        if len(a) and len(b): v.append(a.mean()-b.mean())
    return float(np.mean(v))
cr=[]
for k in range(R):
    srep=sc-cross_val_predict(Ridge(alpha=1.0),ph,sc,cv=KFold(5,shuffle=True,random_state=k))
    cr.append({"fmt_correct":paired(srep,verbose&correct,~verbose&correct),"fmt_buggy":paired(srep,verbose&~correct,~verbose&~correct),"margin_terse":paired(srep,correct&~verbose,~correct&~verbose)})
out["mbpp_code"]={"R":R,"fmt_correct_range":[min(r["fmt_correct"] for r in cr),max(r["fmt_correct"] for r in cr)],"fmt_buggy_range":[min(r["fmt_buggy"] for r in cr),max(r["fmt_buggy"] for r in cr)],"margin_terse_range":[min(r["margin_terse"] for r in cr),max(r["margin_terse"] for r in cr)]}
print("mbpp",out["mbpp_code"],flush=True)
json.dump(out,open(os.path.join(HERE,"repeat_fit_stability.json"),"w"),indent=1); print("wrote")
