#!/usr/bin/env python3
"""Deployment-rule search with scale-matched switches (v171 review, M8).
Same DEV/HELD protocol as obs_rule_search.py, grouped cross-fitting throughout, and three
additions the review asked for: (i) switches whose residual branch is rescaled to the raw
score's scale on the training folds, (ii) a calibrated blend of z-scored raw and residual,
(iii) a query-conditioned adjustment (subtract the within-query-centered surface component).
Full-population AUC delta over raw, query-clustered CIs on held-out sets."""
import json, os, sys, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pr_common import load, features, HERE
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
def objects(name):
    q,ans,y,s,qid,_=load(name); s=np.asarray(s,float); dense,phi=features(q,ans); cv=GroupKFold(5)
    g=np.zeros_like(s); zrep=np.zeros_like(s); mu_s=np.zeros_like(s); sd_s=np.ones_like(s); c=np.zeros_like(s)
    for tr,te in cv.split(phi,s,groups=qid):
        m=Ridge(alpha=1.0).fit(phi[tr],s[tr]); g[te]=m.predict(phi[te])
        # training-fold scale of raw and residual (residual computed in-sample on train for scale only)
        r_tr=s[tr]-m.predict(phi[tr]); mu_s[te]=s[tr].mean(); sd_s[te]=s[tr].std()
        zrep[te]=(s[te]-g[te]-r_tr.mean())/(r_tr.std()+1e-9)*s[tr].std()+s[tr].mean()   # residual mapped to raw scale
        art=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(dense[tr],y[tr]); c[te]=art.decision_function(dense[te])
    return {"y":y,"s":s,"g":g,"srep":s-g,"zrep":zrep,"c":c,"qid":qid}
def apply(o,fam,p):
    s,g,c,zrep,qid=o["s"],o["g"],o["c"],o["zrep"],o["qid"]
    if fam=="F2r":   # |g|-switch, residual rescaled on training folds
        t=np.percentile(np.abs(g-np.median(g)),p); m=np.abs(g-np.median(g))>t; out=s.copy(); out[m]=zrep[m]; return out
    if fam=="F3r":   # c-band switch, rescaled
        lo,hi=np.percentile(c,p),np.percentile(c,100-p); m=(c<lo)|(c>hi); out=s.copy(); out[m]=zrep[m]; return out
    if fam=="Blend": # calibrated blend of z-scored raw and residual
        zs=(s-s.mean())/s.std(); zr=(o["srep"]-o["srep"].mean())/o["srep"].std(); return (1-p)*zs+p*zr
    if fam=="QC":    # query-conditioned: subtract within-query-centered g, scaled by p
        gc=g.copy()
        for qq in np.unique(qid):
            i=qid==qq; gc[i]=g[i]-g[i].mean()
        return s-p*gc
GRID={"F2r":[50,70,80,90],"F3r":[10,20,30],"Blend":[0.25,0.5,0.75],"QC":[0.5,1.0]}
DEV=["wikiqa","asnq"]; HELD=["squad","duorc"]
objs={n:objects(n) for n in DEV+HELD+["triviaqa"]}; print("objects built",flush=True)
def cluster_ci(o,u,B=500):
    qs=np.unique(o["qid"]); gains=[]
    for _ in range(B):
        take=rng.choice(qs,len(qs),replace=True); idx=np.concatenate([np.where(o["qid"]==t)[0] for t in take]); ya=o["y"][idx]
        if ya.min()==ya.max(): continue
        gains.append(roc_auc_score(ya,u[idx])-roc_auc_score(ya,o["s"][idx]))
    return [float(np.percentile(gains,2.5)),float(np.percentile(gains,97.5))]
res={"dev_selection":{},"frozen_heldout":{}}; frozen={}
for fam,grid in GRID.items():
    sc=[(float(np.mean([roc_auc_score(objs[n]["y"],apply(objs[n],fam,p))-roc_auc_score(objs[n]["y"],objs[n]["s"]) for n in DEV])),p) for p in grid]
    best=max(sc); frozen[fam]=best[1]; res["dev_selection"][fam]={"grid":[(p,round(d,4)) for d,p in sc],"selected":best[1],"dev_mean_delta":round(best[0],4)}
    print(fam,res["dev_selection"][fam],flush=True)
for n in HELD+["triviaqa"]:
    o=objs[n]; row={}
    for fam,p in frozen.items():
        u=apply(o,fam,p); d=float(roc_auc_score(o["y"],u)-roc_auc_score(o["y"],o["s"])); row[fam]={"param":p,"full_delta":round(d,4),"clusterCI":[round(x,4) for x in cluster_ci(o,u)]}
    res["frozen_heldout"][n]=row; print(n,row,flush=True)
json.dump(res,open(os.path.join(HERE,"rule_search_rescaled.json"),"w"),indent=1); print("wrote")
