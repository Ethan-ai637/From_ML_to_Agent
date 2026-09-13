import numpy as np
rng=np.random.default_rng(2)
T,d=6,4
Q=rng.normal(size=(T,d)); K=rng.normal(size=(T,d)); V=rng.normal(size=(T,d))
S=Q@K.T/np.sqrt(d); S[np.triu_indices(T,1)]=-1e9
A=np.exp(S-S.max(axis=1,keepdims=True)); A/=A.sum(axis=1,keepdims=True)
O=A@V
gate=1/(1+np.exp(-rng.normal(size=(T,1))))
print('attention=\n',np.round(A,3))
print('output norm',np.linalg.norm(O),'gated norm',np.linalg.norm(gate*O))
