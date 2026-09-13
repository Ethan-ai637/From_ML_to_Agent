import numpy as np, matplotlib.pyplot as plt, heapq
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'; out.mkdir(parents=True,exist_ok=True); N=12; start=(0,0); goal=(N-1,N-1)
def nbr(x):
 i,j=x
 for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
  y=(i+di,j+dj)
  if 0<=y[0]<N and 0<=y[1]<N: yield y
def astar(blocked,source=start):
 pq=[(0,0,source)]; g={source:0}; par={source:None}
 while pq:
  _,gc,u=heapq.heappop(pq)
  if gc!=g[u]: continue
  if u==goal:
   p=[]
   while u is not None: p.append(u); u=par[u]
   return p[::-1]
  for v in nbr(u):
   if v in blocked: continue
   ng=gc+1
   if ng<g.get(v,999): g[v]=ng; par[v]=u; heapq.heappush(pq,(ng+abs(goal[0]-v[0])+abs(goal[1]-v[1]),ng,v))
 return None
def episode(pfail,replan,static,unif):
 path=astar(static); pos=start; disc=set(static); idx=1; ui=0; steps=0
 if path is None:return False
 while pos!=goal and steps<5*N*N:
  p=astar(disc,pos) if replan else path[idx:]
  if not p or (replan and len(p)<2): return False
  nxt=p[1] if replan else path[idx]; u=unif[ui]; ui+=1
  if u<pfail and nxt!=goal: disc.add(nxt); steps+=1; 
  else: pos=nxt; steps+=1; idx+=0 if replan else 1
  if nxt in disc and not replan:return False
 return pos==goal
ps=np.linspace(0,.18,7); oo=[]; cc=[]
for p in ps:
 a=[]; b=[]
 for trial in range(240):
  rng=np.random.default_rng(10000+trial); static={(i,j) for i in range(N) for j in range(N) if (i,j) not in (start,goal) and rng.random()<.08}; unif=rng.random(5*N*N); a.append(episode(p,False,static,unif)); b.append(episode(p,True,static,unif))
 oo.append(np.mean(a)); cc.append(np.mean(b))
plt.figure(figsize=(7.2,4.4)); plt.plot(ps,oo,marker='o',label='open-loop'); plt.plot(ps,cc,marker='s',label='replan'); plt.legend(); plt.tight_layout(); plt.savefig(out/'ch21_planning_tools.pdf')
