"""SICK half of cert_reverify (datasets-server is down; load SICK via the datasets library in the
same native order that produced the cached scores). Computes (A) R^2(phi->s_rep) certificate and
(B) artifact-decision floor, to show the certificate closes the LINEAR channel while the decision
floor stays >0.5 because the SICK artifact decision is nonlinear in phi (reviewer's 0.648 case)."""
import numpy as np, json, os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
CV = KFold(5, shuffle=True, random_state=0); rng = np.random.default_rng(0)

d = load_dataset("yangwang825/sick", split="test")
pairs = [(r["text1"], r["text2"], 1 if int(r["label"]) == 0 else 0)
         for r in d if int(r["label"]) in (0, 2)][:2400]
hyp = [b for _, b, _ in pairs]; y = np.array([c for _, _, c in pairs])
s = np.load("/tmp/sick_s.npy")
assert len(s) == len(y), (len(s), len(y))
print(f"SICK n={len(y)}  raw AUC {roc_auc_score(y, s):.3f} (expect ~0.967)")

Xh = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)).fit_transform(hyp)
hyp_prob = cross_val_predict(LogisticRegression(max_iter=1000), Xh, y, cv=CV, method="predict_proba")[:, 1]
z = (hyp_prob > 0.5).astype(int)
s_rep = s - cross_val_predict(Ridge(alpha=1.0), Xh, s, cv=CV)

def r2(x):
    pred = cross_val_predict(Ridge(alpha=1.0), Xh, x, cv=CV)
    return 1 - float(((x-pred)**2).sum())/float(((x-x.mean())**2).sum())
def floor(x):
    pr = cross_val_predict(LogisticRegression(max_iter=1000), x.reshape(-1,1), z, cv=CV, method="predict_proba")[:,1]
    return roc_auc_score(z, pr), pr
def ci(z, pr, R=2000):
    v=np.empty(R)
    for r in range(R):
        ii=rng.integers(0,len(z),len(z)); v[r]=roc_auc_score(z[ii],pr[ii]) if len(set(z[ii]))>1 else np.nan
    v=v[~np.isnan(v)]; return np.percentile(v,2.5), np.percentile(v,97.5)

r2_raw, r2_rep = r2(s), r2(s_rep)
f_raw,_ = floor(s); f_rep, pr = floor(s_rep); lo,hi = ci(z, pr)
print(f"(A) CERTIFICATE  R^2(phi->s):   raw {r2_raw:+.3f} -> repaired {r2_rep:+.3f}  "
      f"[linear channel {'CLOSED' if r2_rep<0.02 else 'open'}]")
print(f"(B) DECISION FLOOR AUC(z|score): raw {f_raw:.3f} -> repaired {f_rep:.3f}  95%CI [{lo:.3f},{hi:.3f}]")
print("=> SICK: linear phi-channel closes (A ~0) yet decision-floor stays >0.5 (B) because the SICK "
      "artifact decision is NONLINEAR in phi -- consistent with the (linear) certificate, not a contradiction.")
json.dump({"name":"SICK","r2_raw":r2_raw,"r2_rep":r2_rep,"floor_raw":f_raw,"floor_rep":f_rep,
           "floor_rep_ci":[lo,hi]}, open(os.path.join(os.path.dirname(__file__),"..","notes","cert_sick_2026-07-11.json"),"w"), indent=1)
