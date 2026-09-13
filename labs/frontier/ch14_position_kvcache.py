import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch14_position_kvcache.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(14); T,d=10,8; X=rng.normal(size=(T,d)); Wq=rng.normal(size=(d,d))/np.sqrt(d); Wk=rng.normal(size=(d,d))/np.sqrt(d); Wv=rng.normal(size=(d,d))/np.sqrt(d)
def sm(z): z-=z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
def attn(X): Q,K,V=X@Wq,X@Wk,X@Wv; A=sm(Q@K.T/np.sqrt(d)); return A@V
Y=attn(X); perm=rng.permutation(T); print('equivariance error',np.max(np.abs(attn(X[perm])-Y[perm])))
def rot(v,a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s],[s,c]])@v
q=np.array([1.2,-.7]); k=np.array([.4,1.1]); ds=np.arange(-20,21); plt.figure(figsize=(7.4,4.5))
for shift in (0,7,19): plt.plot(ds,[rot(q,(shift+30)*.37)@rot(k,(shift+30+dd)*.37) for dd in ds],label=f'shift={shift}')
plt.legend(); plt.tight_layout(); plt.savefig(out)
