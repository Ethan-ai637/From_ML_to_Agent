import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
x=np.linspace(-2,2,20); y=np.sin(2*x)+0.15*rng.normal(size=len(x))
xx=np.linspace(-2.5,2.5,400); yy=np.sin(2*xx)
for deg in [3,10,18]:
    X=np.vander(x,deg+1); w=np.linalg.lstsq(X,y,rcond=None)[0]
    pred=np.vander(xx,deg+1)@w
    train=np.mean((X@w-y)**2); test=np.mean((pred-yy)**2)
    plt.plot(xx,pred,label=f'deg={deg}, train={train:.3f}, test={test:.3f}')
plt.scatter(x,y); plt.plot(xx,yy,linestyle='--',label='truth'); plt.legend(); plt.show()
