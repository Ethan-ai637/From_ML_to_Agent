import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
rng=np.random.default_rng(19); out=Path(__file__).resolve().parents[2]/'figures'/'generated'; out.mkdir(parents=True,exist_ok=True)
def trial(budget=3,nc=12,nd=12):
 tc=rng.integers(nc); td=rng.integers(nd); cs=rng.normal(size=nc); cs[tc]+=.6; ds=rng.normal(size=(nc,nd)); ds[tc,td]+=1.; ranks=np.argsort((cs[:,None]+.45*ds).ravel())[::-1][:budget]; one=tc*nd+td in ranks; bridge=cs+rng.normal(0,.45,nc); bridge[tc]+=1.8; cc=int(np.argmax(bridge)); local=np.argsort(ds[cc])[::-1][:max(1,budget-1)]; return one, cc==tc and td in local
bs=np.arange(2,9); aa=[]; bb=[]
for b in bs:
 r=[trial(int(b)) for _ in range(6000)]; aa.append(np.mean([x for x,y in r])); bb.append(np.mean([y for x,y in r]))
plt.figure(figsize=(7.2,4.4)); plt.plot(bs,aa,marker='o',label='one-shot'); plt.plot(bs,bb,marker='s',label='adaptive'); plt.legend(); plt.tight_layout(); plt.savefig(out/'ch19_agentic_retrieval.pdf')
