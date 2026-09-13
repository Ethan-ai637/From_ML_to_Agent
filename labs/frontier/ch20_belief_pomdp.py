import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'; out.mkdir(parents=True,exist_ok=True); gamma=.95; ph=.85; beliefs=np.linspace(.001,.999,999); V=np.zeros_like(beliefs)
def interp(p,v): return np.interp(np.clip(p,beliefs[0],beliefs[-1]),beliefs,v)
for it in range(500):
 ql=beliefs*(-100)+(1-beliefs)*10; qr=beliefs*10+(1-beliefs)*(-100); po=beliefs*ph+(1-beliefs)*(1-ph); pl=beliefs*ph/po; pr=beliefs*(1-ph)/(1-po); q=-1+gamma*(po*interp(pl,V)+(1-po)*interp(pr,V)); nv=np.maximum.reduce([ql,qr,q]);
 if np.max(np.abs(nv-V))<1e-10: V=nv; break
 V=nv
plt.figure(figsize=(7.2,4.4)); plt.plot(beliefs,ql,label='open left'); plt.plot(beliefs,qr,label='open right'); plt.plot(beliefs,q,label='listen'); plt.legend(); plt.tight_layout(); plt.savefig(out/'ch20_belief_pomdp.pdf')
