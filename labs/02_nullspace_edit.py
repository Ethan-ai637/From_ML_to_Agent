import numpy as np
rng=np.random.default_rng(1)
d=8; k=3
H=rng.normal(size=(d,k))
U,_,_=np.linalg.svd(H,full_matrices=False)
Pnull=np.eye(d)-U@U.T
E=rng.normal(size=(d,d))
E_safe=E@Pnull
print('raw change on preserved states :',np.linalg.norm(E@H))
print('null-projected change          :',np.linalg.norm(E_safe@H))
print('idempotence error              :',np.linalg.norm(Pnull@Pnull-Pnull))
