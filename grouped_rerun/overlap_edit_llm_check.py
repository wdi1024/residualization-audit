#!/usr/bin/env python3
"""Blinded automatic spot-check of the SQuAD overlap edits (v171 review, M9).
Rebuilds the deterministic edits of overlap_intervention.py and asks an LLM judge, blind to
which sentence is original or edited and to the intended label, two questions per sentence:
(1) does the sentence contain the answer to the question (yes/no), (2) is the sentence fluent
English (1-5). Reports the rate at which the edit preserves the containment judgment and the
fluency change. This is an automatic check, not a human adjudication."""
import json, os, re, sys, time, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pr_common import load, HERE
from nltk.corpus import wordnet as wn
import anthropic
STOP=set("the a an of to in and or is are was were be been on for with as at by from that this it its he she they what which who when where how why did does do".split())
def synonym(word):
    for syn in wn.synsets(word.lower()):
        for lemma in syn.lemmas():
            w=lemma.name().replace("_"," ")
            if w.lower()!=word.lower() and " " not in w and w.isalpha(): return w
    return None
def edit_down(question,sent,answer):
    qw={w.lower() for w in question.split()}-STOP; out=[]; ch=0
    for w in sent.split():
        core=re.sub(r"\W","",w)
        if core.lower() in qw and core.lower() not in answer.lower() and answer not in w:
            syn=synonym(core)
            if syn: out.append(w.replace(core,syn)); ch+=1; continue
        out.append(w)
    return " ".join(out),ch
def edit_up(question,sent):
    kws=[w for w in question.split() if w.lower() not in STOP and re.sub(r"\W","",w).isalpha()][:3]
    return None if not kws else sent.rstrip(".")+", regarding "+" ".join(kws)+"."
q,ans_s,y,s,qid,extra=load("squad"); answers=extra["answers"]; rng=np.random.default_rng(0)
down=[i for i in range(len(q)) if y[i]==1][:600]; up=list(rng.choice([i for i in range(len(q)) if y[i]==0],600,replace=False))
items=[]
for i in down:
    e,ch=edit_down(q[i],ans_s[i],answers[i])
    if ch>=1 and answers[i] in e: items.append(("down",i,e))
for i in up:
    e=edit_up(q[i],ans_s[i])
    if e and answers[i] not in e: items.append(("up",i,e))
rng.shuffle(items); items=items[:120]   # spot-check sample
client=anthropic.Anthropic(); MODEL="claude-haiku-4-5-20251001"
def judge(question,sentence):
    prompt=(f"Question: {question}\nSentence: {sentence}\n\nAnswer in JSON with two fields: \"contains_answer\" (true if the sentence states the answer to the question, false otherwise) and \"fluency\" (1-5, 5 = natural English). Output only the JSON.")
    for k in range(4):
        try:
            r=client.messages.create(model=MODEL,max_tokens=60,temperature=0,messages=[{"role":"user","content":prompt}]); t=r.content[0].text
            m=re.search(r"\{.*\}",t,re.S); j=json.loads(m.group(0)); return bool(j["contains_answer"]),int(j["fluency"])
        except Exception as ex: time.sleep(2*(k+1)); err=ex
    return None,None
recs=[]
for kind,i,e in items:
    # blinded: original and edited are presented as independent items in random order
    pair=[("orig",ans_s[i]),("edit",e)]; rng.shuffle(pair); res={}
    for tag,sent in pair: res[tag]=judge(q[i],sent)
    recs.append({"kind":kind,"i":int(i),"label":int(y[i]),"orig":res["orig"],"edit":res["edit"]})
    if len(recs)%20==0: print(len(recs),flush=True)
def summ(kind):
    rs=[r for r in recs if r["kind"]==kind and r["orig"][0] is not None and r["edit"][0] is not None]
    lab=rs[0]["label"] if rs else None
    return {"n":len(rs),"label":lab,"orig_contains_rate":float(np.mean([r["orig"][0] for r in rs])),"edit_contains_rate":float(np.mean([r["edit"][0] for r in rs])),
            "containment_judgment_preserved":float(np.mean([r["orig"][0]==r["edit"][0] for r in rs])),"fluency_orig":float(np.mean([r["orig"][1] for r in rs])),"fluency_edit":float(np.mean([r["edit"][1] for r in rs]))}
out={"model":MODEL,"n_items":len(recs),"down":summ("down"),"up":summ("up"),"records":recs}
print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=1))
json.dump(out,open(os.path.join(HERE,"overlap_edit_llm_check.json"),"w"),indent=1); print("wrote")
