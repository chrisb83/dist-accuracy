import mpmath as mp
from vref import *
for d in (50,100):
    mp.mp.dps=d
    print(d, mp.nstr(mp.mpf(1)/2 + mid(0,3,100,0.01,0),35))
