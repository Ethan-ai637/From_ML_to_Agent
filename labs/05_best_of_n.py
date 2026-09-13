import numpy as np
for p in [0.02,0.1,0.3,0.6]:
    vals=[1-(1-p)**n for n in [1,2,4,8,16,32,64]]
    print('p=',p,' -> ',[round(v,3) for v in vals])
