from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch04_alignment_elasticity.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
p0=np.array([.52,.22,.14,.08,.04]); pa=np.array([.08,.14,.18,.26,.34]); z=np.log(p0+1e-12)
def softmax(z): q=np.exp(z-z.max()); return q/q.sum()
def step(z,p,lr): return z-lr*(softmax(z)-p)
def ce(p,q): return -np.sum(p*np.log2(q+1e-12))
trace=[]; align=120
for t in range(align): z=step(z,pa,.25); q=softmax(z); trace.append((ce(pa,q),ce(p0,q)))
for t in range(300): z=step(z,p0,.25); q=softmax(z); trace.append((ce(pa,q),ce(p0,q)))
a=np.array(trace); plt.figure(figsize=(7.2,4.3)); plt.plot(a[:,0],label='alignment distribution'); plt.plot(a[:,1],label='pretraining distribution'); plt.axvline(align,ls='--'); plt.ylabel('bits/token'); plt.legend(); plt.tight_layout(); plt.savefig(out)
