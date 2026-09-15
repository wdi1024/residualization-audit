import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
import os
torch.set_num_threads(6)
CV=KFold(n_splits=5,shuffle=True,random_state=0)   # shuffled: SNLI val is order-agnostic but be rigorous
d=load_dataset('stanfordnlp/snli',split='validation')
# binary: entailment(0) vs contradiction(2); y=1 for entailment
rows=[(r['premise'],r['hypothesis'],1 if r['label']==0 else 0) for r in d if r['label'] in (0,2)]
rows=rows[:2400]
prem=[a for a,_,_ in rows]; hyp=[b for _,b,_ in rows]; y=np.array([c for _,_,c in rows])
name="typeform/distilbert-base-uncased-mnli"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForSequenceClassification.from_pretrained(name).eval()
id2=m.config.id2label; ent_idx=[i for i,l in id2.items() if 'entail' in l.lower()][0]
def pent(p,h):
    with torch.no_grad():
        lo=m(**tok(p,h,return_tensors="pt",truncation=True,max_length=256)).logits[0]
        return torch.softmax(lo,-1)[ent_idx].item()
cache="/tmp/snli_s_2400.npy"
if os.path.exists(cache):
    s=np.load(cache); print("loaded cached scores")
else:
    s=[]
    for i,(p,h) in enumerate(zip(prem,hyp)):
        s.append(pent(p,h))
        if i%400==0: print("scored",i,flush=True)
    s=np.array(s); np.save(cache,s)
# artifact phi = hypothesis-only TF-IDF
Xh=TfidfVectorizer(max_features=8000,ngram_range=(1,2)).fit_transform(hyp)
# hypothesis-only model (the artifact channel, scalar) + residualize s on full phi
hyp_prob=cross_val_predict(LogisticRegression(max_iter=1000),Xh,y,cv=CV,method="predict_proba")[:,1]
s_rep=s-cross_val_predict(Ridge(alpha=1.0),Xh,s,cv=CV)
adv=(hyp_prob>0.5)!=(y==1)   # artifact-adversarial slice: hypothesis-only is WRONG
def auc(mask,x): return roc_auc_score(y[mask],x[mask])
print(f"\n=== NLI (SNLI, distilbert-mnli, n={len(y)}) — repair ===")
print(f"hypothesis-only AUC (artifact strength): {roc_auc_score(y,hyp_prob):.3f}")
print(f"{'':10}{'AUC full':>10}{'AUC on artifact-adversarial slice':>36}")
print(f"{'raw s':10}{auc(np.ones(len(y),bool),s):>10.3f}{auc(adv,s):>36.3f}   (n_adv={adv.sum()})")
print(f"{'repaired':10}{auc(np.ones(len(y),bool),s_rep):>10.3f}{auc(adv,s_rep):>36.3f}")
print("기대: 아티팩트-오도 슬라이스에서 raw는 낮고 repaired가 올라가면 = 양성 (복원이 정렬 회복)")
