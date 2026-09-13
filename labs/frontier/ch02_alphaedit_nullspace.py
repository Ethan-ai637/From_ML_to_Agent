from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(7); out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch02_nullspace_edit.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
d,k=64,20; H=rng.normal(size=(d,k)); Q,_=np.linalg.qr(H); P=np.eye(d)-Q@Q.T; E=rng.normal(size=(12,d)); Ep=E@P
pres=Q@rng.normal(size=(k,200)); edit=P@rng.normal(size=(d,200))+.15*Q@rng.normal(size=(k,200)); scales=np.logspace(-2,1,40)
raw=[]; safe=[]; er=[]; es=[]
for s in scales:
 raw.append(np.linalg.norm(s*E@pres,'fro')); safe.append(np.linalg.norm(s*Ep@pres,'fro')); er.append(np.linalg.norm(s*E@edit,'fro')); es.append(np.linalg.norm(s*Ep@edit,'fro'))
plt.figure(figsize=(7.2,4.3)); plt.loglog(er,raw,label='unconstrained'); plt.loglog(es,np.maximum(safe,1e-15),label='null-space projected'); plt.xlabel('edit effect'); plt.ylabel('preservation damage'); plt.legend(); plt.tight_layout(); plt.savefig(out)
print('max preservation residual',max(safe))
