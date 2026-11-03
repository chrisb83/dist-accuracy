import time, sys, mpmath as mp
from vref import *
B=0.9999999999
pts = [('cdf',0,-1,1,1),('cdf',-2,-1,1,1),('cdf',-200,-1,1,1),('sf',10,-1,1,1),
       ('sf',1e4,-1.5,1,B),('sf',0.008,1,1e6,0),('cdf',-0.008,1,1e6,0),('sf',1e6,-1.5,1,B),('sf',1e12,-1.5,1,B),('sf',1e9,-1.5,1,B),
       ('cdf',-0.002,1,1e6,0),('sf',0.0005,1,1e6,0),('sf',1000,-1.5,1,B)]
for dps in (int(sys.argv[1]),):
    mp.mp.dps = dps
    for m,*args in pts:
        t=time.time(); v={'cdf':cdf,'sf':sf}[m](*args)
        print(dps, m, args, mp.nstr(v,35), round(time.time()-t,1), flush=True)
