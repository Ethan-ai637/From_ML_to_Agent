"""Chapter 1 mechanism-level reproduction: two-timescale delayed generalization.
CPU-only. This is NOT an exact paper reproduction; it isolates a spectral mechanism.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch01_grokking_timescales.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
eta=.08; lam=.012; steps=4000
a=np.empty(steps); b=np.empty(steps); a[0]=b[0]=1.
for t in range(steps-1): a[t+1]=(1-eta*(1+lam))*a[t]; b[t+1]=(1-eta*lam)*b[t]
train=.5*a*a; pop=.5*(a*a+b*b); th=1e-2
t1=int(np.argmax(train<th)); t2=int(np.argmax(pop<th))
plt.figure(figsize=(7.2,4.3)); plt.semilogy(train,label='empirical risk'); plt.semilogy(pop,label='population risk'); plt.axvline(t1,ls='--'); plt.axvline(t2,ls=':'); plt.xlabel('gradient steps'); plt.ylabel('risk'); plt.legend(); plt.tight_layout(); plt.savefig(out)
print({'t1':t1,'t2':t2,'gap':t2-t1})
