import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'figures'/'generated'/'ch06_spectral_filters.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
lams=np.array([1.,.03]); eta=.8; steps=np.arange(180); f=np.array([1-(1-eta*lams)**t for t in steps])
plt.figure(figsize=(7.2,4.4)); plt.plot(steps,f[:,0],label='fast'); plt.plot(steps,f[:,1],label='slow'); plt.xlabel('gradient steps'); plt.ylabel('fraction learned'); plt.legend(); plt.tight_layout(); plt.savefig(out)
print([int(np.argmax(f[:,i]>=.5)) for i in range(2)])
