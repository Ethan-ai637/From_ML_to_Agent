import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch16_testtime_compute.pdf'; out.parent.mkdir(parents=True,exist_ok=True); p=np.array([.04,.07,.10,.16,.24,.36,.53,.70]); K=np.arange(1,65); success=1-(1-p[:,None])**K; B=64; alloc=np.ones(len(p),int)
for _ in range(B-len(p)): j=np.argmax((1-p)**alloc*p); alloc[j]+=1
rng=np.random.default_rng(11)
def noisy(pp,n,trials=8000): y=rng.random((trials,n))<pp; s=rng.normal(size=(trials,n))+y; j=np.argmax(s,1); return y[np.arange(trials),j].mean()
no=np.array([noisy(.18,int(k)) for k in K]); perfect=1-(1-.18)**K; fig,ax=plt.subplots(1,2,figsize=(10.5,4)); ax[0].plot(K,success.mean(0)); ax[1].plot(K,perfect,label='perfect verifier'); ax[1].plot(K,no,label='noisy verifier'); ax[1].legend(); fig.tight_layout(); fig.savefig(out); print('allocation',alloc.tolist())
