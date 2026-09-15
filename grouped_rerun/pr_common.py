"""Shared loaders for the v171 review computations (same row construction as grouped_full.py)."""
import json, os, re, numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,"../cache")
def overlap(a,b):
    A,B=set(a.lower().split()),set(b.lower().split()); return len(A&B)/max(1,len(A|B))
def sent_split(t):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+",t) if len(p.strip())>=3]
def _grp(q):
    u={qq:i for i,qq in enumerate(dict.fromkeys(q))}; return np.array([u[qq] for qq in q])
def load(name):
    if name=="wikiqa":
        d=load_dataset("microsoft/wiki_qa",split="train"); rows=[(r["question"],r["answer"],r["label"]) for r in d][:8000]
        q=[a for a,_,_ in rows]; ans=[b for _,b,_ in rows]; y=np.array([c for _,_,c in rows]); return q,ans,y,np.load(f"{CACHE}/wikiqa_ce_s_8000.npy"),_grp(q),None
    if name=="asnq":
        rows=json.load(open(f"{CACHE}/asnq_rows_8000.json")); q=[a for a,_,_ in rows]; ans=[b for _,b,_ in rows]; y=np.array([c for _,_,c in rows]); return q,ans,y,np.load(f"{CACHE}/asnq_ce_s_8000.npy"),_grp(q),None
    if name=="triviaqa":
        rows=np.load(f"{CACHE}/triviaqa_rows.npy",allow_pickle=True); q=[r[0] for r in rows]; ans=[r[1] for r in rows]; y=np.array([int(r[2]) for r in rows]); return q,ans,y,np.load(f"{CACHE}/triviaqa_ce_s.npy"),_grp(q),None
    if name=="squad":
        d=load_dataset("squad",split="validation"); rows,qid,y2,seen,answers=[],[],[],0,[]
        for r in d:
            sents=sent_split(r["context"])
            if not (3<=len(sents)<=25): continue
            ans=r["answers"]["text"][0]; pos=[i for i,s in enumerate(sents) if ans in s]
            if len(pos)!=1: continue
            others=r["answers"]["text"][1:]
            for i,s in enumerate(sents):
                rows.append((r["question"],s,1 if i==pos[0] else 0)); qid.append(seen); answers.append(ans)
                y2.append(1 if (others and any(o in s for o in others)) else (0 if others else -1))
            seen+=1
            if len(rows)>=8000: break
        rows=rows[:8000]; return [r[0] for r in rows],[r[1] for r in rows],np.array([r[2] for r in rows]),np.load(f"{CACHE}/squad_ce_s_8000.npy"),np.array(qid[:8000]),{"y2":np.array(y2[:8000]),"answers":answers[:8000]}
    if name=="duorc":
        d=load_dataset("ibm/duorc","SelfRC",split="validation"); rows,qid,seen=[],[],0
        for r in d:
            sents=sent_split(r["plot"])
            if not (3<=len(sents)<=25) or r["no_answer"] or not r["answers"]: continue
            ans=r["answers"][0]
            if len(ans.split())>12 or not ans.strip(): continue
            pos=[i for i,s in enumerate(sents) if ans in s]
            if len(pos)!=1: continue
            for i,s in enumerate(sents): rows.append((r["question"],s,1 if i==pos[0] else 0)); qid.append(seen)
            seen+=1
            if len(rows)>=8000: break
        rows=rows[:8000]; return [r[0] for r in rows],[r[1] for r in rows],np.array([r[2] for r in rows]),np.load(f"{CACHE}/duorc_ce_s_8000.npy"),np.array(qid[:8000]),None
    if name=="nq":
        rows=json.load(open(f"{CACHE}/nq_rows.json")); q=[r[0] for r in rows]; ans=[r[1] for r in rows]; y=np.array([int(r[2]) for r in rows])
        qid=np.array([int(r[3]) for r in rows]) if len(rows[0])>3 else _grp(q); return q,ans,y,np.load(f"{CACHE}/nq_ce_s.npy"),qid,None
def features(q,ans):
    ov=np.array([overlap(a,b) for a,b in zip(q,ans)])[:,None]; la=np.array([len(x.split()) for x in ans],float)[:,None]; lq=np.array([len(x.split()) for x in q],float)[:,None]
    dense=np.hstack([ov,la,lq,la-lq]); Xa=TfidfVectorizer(max_features=8000,ngram_range=(1,2)).fit_transform(ans)
    return dense,hstack([Xa,csr_matrix(dense)]).tocsr()
