import numpy as np, torch, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6); N=400
ds=load_dataset('allenai/reward-bench', split='filtered').shuffle(seed=1).select(range(N))
name="OpenAssistant/reward-model-deberta-v3-large-v2"
tok=AutoTokenizer.from_pretrained(name); rm=AutoModelForSequenceClassification.from_pretrained(name).eval()
def sc(q,a):
    with torch.no_grad(): return rm(**tok(q,a,return_tensors="pt",truncation=True,max_length=512)).logits[0,0].item()
def feats(t):  # rich artifact features (format + length)
    return [len(t), t.count("\n"), t.count("- ")+t.count("* "), t.count("```"), t.count("#"), len(re.findall(r"\d+\.",t))]
mc=[];mr=[];fc=[];fr=[]
for i,r in enumerate(ds):
    mc.append(sc(r["prompt"],r["chosen"])); mr.append(sc(r["prompt"],r["rejected"]))
    fc.append(feats(r["chosen"])); fr.append(feats(r["rejected"]))
    if i%80==0: print("scored",i,flush=True)
mc,mr=np.array(mc),np.array(mr); fc,fr=np.array(fc,float),np.array(fr,float)
margin=mc-mr; correct=(margin>0).astype(int)
phi=fc-fr                                   # artifact gap (chosen - rejected) per feature
lengap=phi[:,0]
# finer slice: does length/format AGREE or DISAGREE with the correct (chosen) side?
agree = lengap>0     # chosen is longer -> length points to correct answer
print(f"\n=== STRONG-POSITIVE probe: RewardBench(400) x deberta, rich artifacts ===")
print(f"overall RM acc: {correct.mean():.3f}")
print(f"acc | length AGREES with correct (chosen longer, n={agree.sum()}): {correct[agree].mean():.3f}")
print(f"acc | length DISAGREES (rejected longer, n={(~agree).sum()}):      {correct[~agree].mean():.3f}")
print(f"  -> gap = length-bias effect (AGREES higher = RM leans on length)")
# repair: residualize margin on rich artifact phi (cross-fit), re-measure alignment vs 'chosen better'
s=margin.copy(); 
s_rep=s-cross_val_predict(LinearRegression(),phi,s,cv=5)
# construct proxy per example: use an INDEPENDENT signal = does the OTHER length-neutral judge... 
# here y is degenerate(all chosen better); so measure accuracy on the DISAGREE slice before/after repair
def acc_at0(x,mask): return ((x[mask]>0).mean())
# residualized margin loses absolute sign calibration; recenter by matching median on agree slice
s_rep_cal = s_rep - np.median(s_rep) + np.median(s[agree])*0  # keep threshold 0
print(f"\nrepair on DISAGREE slice (length misleads): raw acc {acc_at0(s,~agree):.3f} -> repaired {acc_at0(s_rep,~agree):.3f}")
print(f"corr(margin, length): {np.corrcoef(margin,lengap)[0,1]:.3f}  corr(margin, phi[:,4]=headers): {np.corrcoef(margin,phi[:,4])[0,1]:.3f}")
