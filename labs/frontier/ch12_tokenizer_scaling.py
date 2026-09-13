import collections, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch12_tokenizer_scaling.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
train=('the model learns from data and the model predicts the next token attention connects tokens ')*120; ind=('the model predicts tokens and the learner updates model parameters from data ')*40; outd=('mira-kala-toru zanu-pela-nemi qiva_170 数据模型注意力 ')*55
def syms(text): return [tuple(list(w)+['</w>']) for w in text.split()]
def apply(v,p):
 a,b=p; o=[]
 for s in v:
  n=[]; i=0
  while i<len(s):
   if i+1<len(s) and s[i]==a and s[i+1]==b: n.append(a+b); i+=2
   else: n.append(s[i]); i+=1
  o.append(tuple(n))
 return o
def merges(text,n):
 v=syms(text); ms=[]
 for _ in range(n):
  c=collections.Counter((a,b) for s in v for a,b in zip(s[:-1],s[1:]));
  if not c: break
  p=c.most_common(1)[0][0]; ms.append(p); v=apply(v,p)
 return ms
def count(text,ms):
 v=syms(text)
 for p in ms: v=apply(v,p)
 return sum(map(len,v))
g=np.arange(0,121,10); ri=[]; ro=[]
for m in g:
 ms=merges(train,int(m)); ri.append(count(ind,ms)/sum(len(w)+1 for w in ind.split())); ro.append(count(outd,ms)/sum(len(w)+1 for w in outd.split()))
plt.figure(figsize=(7.4,4.5)); plt.plot(g,ri,marker='o',label='in-domain'); plt.plot(g,ro,marker='s',label='shifted'); plt.legend(); plt.tight_layout(); plt.savefig(out)
