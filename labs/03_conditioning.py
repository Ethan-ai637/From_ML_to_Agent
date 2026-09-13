import numpy as np
A=np.diag([1.,100.]); w=np.array([3.,3.]); eta=0.018
for t in range(20):
    if t%2==0: print(t,w,0.5*w@A@w)
    w=w-eta*A@w
