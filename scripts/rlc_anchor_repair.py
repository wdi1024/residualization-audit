"""Real-data repair on RLC XSTest-450 (unrepairable anchor). Run from
agents/results/representation_label_coupling/. Transparent TF-IDF router (not the
paper's exact router). OLS residualization; upgrade to LEACE for full erasure."""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
b="data/"
tr=json.load(open(b+"phase3_xstest_full_traces_qwen3.5-2b.json"))["records"]
jd=json.load(open(b+"phase3_xstest_full_judge_qwen3.5-2b_openai_gpt-4o-mini.json"))["records"]
lab={r["id"]:(int(bool(r["refusal_judge"])),int(bool(r["refusal_keyword"]))) for r in jd}
ids=[r["id"] for r in tr if r["id"] in lab]
traces=[next(x["trace"] for x in tr if x["id"]==i) for i in ids]
y=np.array([lab[i][0] for i in ids]); z=np.array([lab[i][1] for i in ids])
Xs=TfidfVectorizer(max_features=20000,ngram_range=(1,2)).fit_transform(traces)
s=cross_val_predict(LogisticRegression(max_iter=1000,class_weight="balanced"),Xs,z,cv=5,method="predict_proba")[:,1]
Xphi=TfidfVectorizer(max_features=5000,ngram_range=(1,3),analyzer="char_wb").fit_transform([t[:50] for t in traces]).toarray()
s_rep=s-cross_val_predict(LinearRegression(),Xphi,s,cv=5)
a=lambda t,x: roc_auc_score(t,x)
print(f"raw     : proxy {a(z,s):.3f}  construct {a(y,s):.3f}")
print(f"repaired: proxy {a(z,s_rep):.3f}  construct {a(y,s_rep):.3f}")
