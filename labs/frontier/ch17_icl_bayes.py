import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch17_icl_bayes.pdf'; out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(17); sigma=.7; mvals=np.arange(21); trials=5000; acc=[]
for m in mvals:
 good=0
 for _ in range(trials):
  s=rng.choice([-1.,1.]); x=rng.normal(size=m); y=s*x+rng.normal(scale=sigma,size=m); llr=-np.sum((y-x)**2)/(2*sigma**2)+np.sum((y+x)**2)/(2*sigma**2); good+=(1. if llr>=0 else -1.)==s
 acc.append(good/trials)
scale=np.linspace(0,10,200); latent=1/(1+np.exp(-(scale-5)/1.35)); fig,ax=plt.subplots(1,2,figsize=(10.5,4)); ax[0].plot(mvals,acc,marker='o'); ax[1].plot(scale,latent,label='latent'); ax[1].plot(scale,(latent>.72).astype(float),ls='--',label='thresholded'); ax[1].legend(); fig.tight_layout(); fig.savefig(out)
