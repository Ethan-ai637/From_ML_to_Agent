from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(3); out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch03_twisted_smc.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
T=18; beta=5.; N=256; trials=120
def p1(prev): return .32 if prev==0 else .42
def rew(k): return 1. if k>=11 else 0.
psi=np.ones((T+1,T+1,2))
for k in range(T+1): psi[T,k,:]=np.exp(beta*rew(k))
for t in range(T-1,-1,-1):
 for k in range(t+1):
  for prev in (0,1):
   p=p1(prev); psi[t,k,prev]=(1-p)*psi[t+1,k,0]+p*psi[t+1,k+1,1]
def ancestral():
 succ=0
 for _ in range(N):
  prev=k=0
  for t in range(T): x=rng.random()<p1(prev); prev=int(x); k+=int(x)
  succ+=rew(k)
 return succ/N
def twisted():
 prev=np.zeros(N,dtype=int); k=np.zeros(N,dtype=int)
 for t in range(T):
  p=np.array([p1(int(v)) for v in prev]); q0=(1-p)*psi[t+1,k,0]; q1=p*psi[t+1,k+1,1]; x=rng.random(N)<q1/(q0+q1); k+=x; prev=x.astype(int)
 return np.mean(k>=11)
a=np.array([ancestral() for _ in range(trials)]); b=np.array([twisted() for _ in range(trials)])
plt.figure(figsize=(7.2,4.3)); plt.hist(a,bins=20,alpha=.55,label='ancestral'); plt.hist(b,bins=20,alpha=.55,label='twisted'); plt.legend(); plt.tight_layout(); plt.savefig(out); print(a.mean(),b.mean())
