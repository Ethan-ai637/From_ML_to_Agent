import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch10_lowrank_momentum.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(10)
m=n=64; r=6; U,_=np.linalg.qr(rng.normal(size=(m,r))); V,_=np.linalg.qr(rng.normal(size=(n,r))); M=np.zeros((m,n))
for t in range(160): G=U@np.diag(rng.normal(size=r))@V.T+.015*rng.normal(size=(m,n)); M=.95*M+.05*G
s=np.linalg.svd(M,compute_uv=False); energy=np.cumsum(s*s)/np.sum(s*s); ranks=np.arange(1,25); errs=np.sqrt(1-energy[ranks-1]); plt.figure(figsize=(7.2,4.4)); plt.plot(ranks,errs,marker='o'); plt.axvline(r,ls='--'); plt.tight_layout(); plt.savefig(out)
