import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch08_feature_learning.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(8)
base=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float); y0=np.array([0,1,1,0]); X=np.repeat(base,60,axis=0)+.28*rng.normal(size=(240,2)); y=np.repeat(y0,60); Y=np.eye(2)[y]; H=8
W1=.45*rng.normal(size=(2,H)); b1=np.zeros(H); W2=.45*rng.normal(size=(H,2)); b2=np.zeros(2); losses=[]; sep=[]
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
for t in range(1200):
 z=X@W1+b1; h=np.maximum(z,0); p=softmax(h@W2+b2); losses.append(-np.mean(np.log(p[np.arange(len(y)),y]+1e-12)))
 if t%20==0: sep.append(np.linalg.norm(h[y==0].mean(0)-h[y==1].mean(0))/(h.std()+1e-8))
 dl=(p-Y)/len(y); dW2=h.T@dl; db2=dl.sum(0); dz=(dl@W2.T)*(z>0); W1-=.06*(X.T@dz); b1-=.06*dz.sum(0); W2-=.06*dW2; b2-=.06*db2
plt.figure(figsize=(7.2,4.4)); plt.plot(losses,label='cross-entropy'); plt.twinx().plot(np.arange(len(sep))*20,sep,ls='--'); plt.tight_layout(); plt.savefig(out)
