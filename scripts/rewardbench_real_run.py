import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
N=300
ds=load_dataset('allenai/reward-bench', split='filtered').shuffle(seed=0).select(range(N))
name="OpenAssistant/reward-model-deberta-v3-large-v2"
tok=AutoTokenizer.from_pretrained(name); rm=AutoModelForSequenceClassification.from_pretrained(name).eval()
def score(q,a):
    with torch.no_grad():
        return rm(**tok(q,a,return_tensors="pt",truncation=True,max_length=512)).logits[0,0].item()
mc=[];mr=[];lc=[];lr=[]
for i,r in enumerate(ds):
    mc.append(score(r["prompt"],r["chosen"])); mr.append(score(r["prompt"],r["rejected"]))
    lc.append(len(r["chosen"])); lr.append(len(r["rejected"]))
    if i%50==0: print("scored",i,flush=True)
mc,mr,lc,lr=map(np.array,(mc,mr,lc,lr))
margin=mc-mr; lengap=lc-lr
acc=(margin>0).mean()
len_floor=((lengap>0)==(margin>0)).mean()          # does longer==RM-preferred coincide?
len_predicts_chosen=(lengap>0).mean()               # how often chosen is the longer one (true-pref side)
# repair proxy: RM accuracy on LENGTH-BALANCED pairs (|lengap| in bottom tercile) vs length-driven pairs
q1=np.quantile(np.abs(lengap),0.33)
bal=np.abs(lengap)<=q1; big=np.abs(lengap)>np.quantile(np.abs(lengap),0.66)
print(f"\n=== REAL RewardBench ({N}) x {name.split('/')[-1]} ===")
print(f"RM accuracy (P margin>0):            {acc:.3f}")
print(f"'chosen is longer' rate:             {len_predicts_chosen:.3f}  (0.5=length neutral)")
print(f"corr(margin, length-gap):            {np.corrcoef(margin,lengap)[0,1]:.3f}")
print(f"RM acc on LENGTH-BALANCED pairs:     {acc if bal.sum()==0 else (margin[bal]>0).mean():.3f}  (n={bal.sum()})")
print(f"RM acc on LENGTH-DOMINATED pairs:    {(margin[big]>0).mean():.3f}  (n={big.sum()})")
print("해석: 길이균형 pairs에서 정확도 떨어지면 = RM이 length 아티팩트에 의존 (repair가 걷어낼 대상)")
