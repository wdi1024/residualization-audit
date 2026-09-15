import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0); N=4000
def run(rho):  # rho = construct-artifact collinearity
    z=rng.normal(size=N)                      # construct latent
    a=rho*z+np.sqrt(1-rho**2)*rng.normal(size=N)  # artifact latent, corr rho with z
    phi=np.c_[a+0.1*rng.normal(size=(N,)), a**2+0.1*rng.normal(size=(N,)), rng.normal(size=N)]  # content-blind artifact features
    s=1.0*z+1.2*a+0.3*rng.normal(size=N)      # eval score = construct + artifact + noise
    ylab=(z+0.3*rng.normal(size=N)>0).astype(int)   # construct label (independent of artifact)
    # repair: residualize s on phi (cross-fitted)
    ghat=cross_val_predict(LinearRegression(), phi, s, cv=5)
    s_rep=s-ghat
    def align(x): return roc_auc_score(ylab, x)          # construct alignment (AUC vs construct label)
    def floor(x):                                        # content-blind floor: predict artifact-sign from score
        alab=(a>0).astype(int)
        return roc_auc_score(alab, cross_val_predict(LogisticRegression(max_iter=500), x.reshape(-1,1), alab, cv=5, method="predict_proba")[:,1])
    return align(s), align(s_rep), floor(s), floor(s_rep)
print(f"{'rho(z,a)':>9} | {'align raw':>9} {'align rep':>9} | {'floor raw':>9} {'floor rep':>9} | verdict")
for rho in [0.0,0.3,0.6,0.9]:
    ar,arep,fr,frep=run(rho)
    v="REPAIR HELPS" if arep>=ar-0.005 else "repair HURTS (collinear)"
    print(f"{rho:>9.1f} | {ar:>9.3f} {arep:>9.3f} | {fr:>9.3f} {frep:>9.3f} | {v}")
print("\n기대: 저공선(rho낮음)=repair가 construct align 유지/개선하며 floor를 chance로; 고공선=align 하락(unrepairable).")
