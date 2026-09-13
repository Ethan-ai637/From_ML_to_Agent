import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch09_gradient_influence.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(9)
N,D,C=80,6,3; centers=rng.normal(size=(C,D)); y=rng.integers(0,C,N); X=centers[y]+.7*rng.normal(size=(N,D)); W=.2*rng.normal(size=(D,C))
def sm(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
def grad(i,W): p=sm(X[i]@W); L=-np.log(p[y[i]]+1e-12); p[y[i]]-=1; return L,np.outer(X[i],p)
_,gi=grad(0,W); eta=2e-3; W2=W-eta*gi; pred=[]; actual=[]
for j in range(1,N): Lj,gj=grad(j,W); Lj2,_=grad(j,W2); pred.append(-eta*np.sum(gi*gj)); actual.append(Lj2-Lj)
plt.figure(figsize=(6.2,5)); plt.scatter(pred,actual,s=20); lo=min(min(pred),min(actual)); hi=max(max(pred),max(actual)); plt.plot([lo,hi],[lo,hi]); plt.tight_layout(); plt.savefig(out); print(np.corrcoef(pred,actual)[0,1])
