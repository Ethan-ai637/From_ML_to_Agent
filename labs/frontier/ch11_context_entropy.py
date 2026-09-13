import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch11_context_entropy.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(11)
N=220000; x=rng.integers(0,2,N,dtype=np.int8)
for t in range(3,N): x[t]=x[t-3]^(rng.random()<.08)
def H(k):
 counts={}
 for t in range(max(k,5000),len(x)):
  ctx=tuple(int(v) for v in x[t-k:t]) if k else (); counts.setdefault(ctx,[0,0])[int(x[t])]+=1
 total=sum(map(sum,counts.values())); ans=0
 for c0,c1 in counts.values():
  n=c0+c1
  for c in (c0,c1):
   if c: p=c/n; ans-=(n/total)*p*np.log2(p)
 return ans
ks=np.arange(9); hs=np.array([H(int(k)) for k in ks]); plt.figure(figsize=(7.4,4.5)); plt.plot(ks,hs,marker='o'); plt.axvline(3,ls='--'); plt.xlabel('context length'); plt.ylabel('conditional entropy'); plt.tight_layout(); plt.savefig(out)
