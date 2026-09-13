import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch18_dpo_rlvr.pdf'; out.parent.mkdir(parents=True,exist_ok=True); p0=np.array([.48,.27,.18,.07]); r=np.array([0.,.3,.9,1.]); betas=np.array([2.,1.,.5,.25,.12]); pol=[]
for b in betas: w=p0*np.exp(r/b); pol.append(w/w.sum())
pol=np.array(pol); p=p0.copy(); hist=[p.copy()]
for _ in range(30): adv=r-p@r; p*=np.exp(.22*adv); p/=p.sum(); hist.append(p.copy())
h=np.array(hist); ent=-np.sum(h*np.log(h+1e-12),axis=1); fig,ax=plt.subplots(1,2,figsize=(10.5,4));
for j in range(4): ax[0].plot(1/betas,pol[:,j],marker='o')
ax2=ax[1].twinx(); ax[1].plot(h[:,2]+h[:,3]); ax2.plot(ent,ls='--'); fig.tight_layout(); fig.savefig(out)
