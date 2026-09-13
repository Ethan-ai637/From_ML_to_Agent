import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
rng=np.random.default_rng(22); out=Path(__file__).resolve().parents[2]/'figures'/'generated'; out.mkdir(parents=True,exist_ok=True); caps=np.array([8,12,16,24,32,48,64]); S={'FIFO':[],'Random':[],'Utility-aware':[]}; T=120
for cap in caps:
 vals={k:[] for k in S}
 for _ in range(1200):
  h=rng.beta(.8,3,T); q=int(rng.choice(T,p=h/h.sum())); sets=[set(range(max(0,T-cap),T)),set(rng.choice(T,cap,replace=False)),set(np.argsort(h)[-cap:])]
  for k,s in zip(S,sets): vals[k].append(q in s)
 for k in S:S[k].append(np.mean(vals[k]))
plt.figure(figsize=(7.2,4.4)); [plt.plot(caps,v,marker='o',label=k) for k,v in S.items()]; plt.legend(); plt.tight_layout(); plt.savefig(out/'ch22_memory_budget.pdf')
T2=4000; eta=np.sqrt(2*np.log(2)/T2); w1=np.array([1.8,.7]); w2=np.array([.6,1.7]); A=np.array([[1.,-1.],[-1.,1.]]); cum=np.zeros(2); real=0.; reg=[]
for t in range(1,T2+1): p=w1/w1.sum(); q=w2/w2.sum(); u=A@q; v=-(p@A); real+=p@A@q; cum+=u; w1*=np.exp(eta*u); w2*=np.exp(eta*v); reg.append((cum.max()-real)/t)
plt.figure(figsize=(7.2,4.4)); plt.plot(np.arange(1,T2+1),reg); plt.tight_layout(); plt.savefig(out/'ch22_selfplay_regret.pdf')
