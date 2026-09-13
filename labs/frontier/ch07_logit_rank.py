import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch07_logit_rank.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(7)
P,R,r=80,60,4; L=rng.normal(size=(P,r))@rng.normal(size=(R,r)).T+.2*rng.normal(size=(P,R)); s=np.linalg.svd(L,compute_uv=False)
plt.figure(figsize=(7.2,4.4)); plt.semilogy(np.arange(1,21),s[:20]/s[0],marker='o'); plt.axvline(r,ls='--'); plt.xlabel('singular-value index'); plt.ylabel('normalized singular value'); plt.tight_layout(); plt.savefig(out)
