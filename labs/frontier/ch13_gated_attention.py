import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch13_gated_attention.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(13); T,d=32,16; Q=rng.normal(size=(T,d)); K=rng.normal(size=(T,d)); V=rng.normal(size=(T,d)); bs=np.linspace(0,10,31); sink=[]; raw=[]; gated=[]
for b in bs:
 S=Q@K.T/np.sqrt(d); S[:,0]+=b; S-=S.max(1,keepdims=True); A=np.exp(S); A/=A.sum(1,keepdims=True); O=A@V; g=1/(1+np.exp(10*(A[:,0]-.55))); sink.append(A[:,0].mean()); raw.append(np.linalg.norm(O,axis=1).mean()); gated.append(np.linalg.norm(g[:,None]*O,axis=1).mean())
plt.figure(figsize=(7.4,4.5)); plt.plot(bs,sink,label='sink mass'); plt.plot(bs,raw,label='ungated norm'); plt.plot(bs,gated,label='gated norm'); plt.legend(); plt.tight_layout(); plt.savefig(out)
