from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(5); out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch05_polar_iteration.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
G=rng.normal(size=(48,32)); U,_,Vt=np.linalg.svd(G,full_matrices=False); Q=U@Vt; X=G/(np.linalg.norm(G,2)+1e-12); errs=[]; orth=[]
for k in range(16): errs.append(np.linalg.norm(X-Q,'fro')/np.linalg.norm(Q,'fro')); orth.append(np.linalg.norm(X.T@X-np.eye(X.shape[1]),'fro')); X=1.5*X-.5*X@(X.T@X)
plt.figure(figsize=(7.2,4.3)); plt.semilogy(errs,marker='o',label='polar error'); plt.semilogy(orth,marker='s',label='orthogonality'); plt.legend(); plt.tight_layout(); plt.savefig(out)
