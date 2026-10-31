import sys; sys.path.insert(0, '.')
import mpmath as mp, numpy as np
from scipy import stats
from vref import *
cases = [('cdf',-0.001,1e4),('cdf',-0.0002,1e4),('cdf',0.0,1e6),('cdf',-0.3,100),('cdf',-3.0,10),
 ('cdf',-9.0,4),('cdf',-3.2,1),('cdf',-6.0,1e-8),('cdf',-1.25,4),('sf',9.0,-4),('sf',1.25,-4),
 ('logpdf',1.5e154,4),('logpdf',-1.5e154,-4),('logpdf',-1.5e154,1e-8),
 ('cdf',-0.033,100),('cdf',-0.0001,1e4),('cdf',0.0,1e5),('cdf',-3.0,1)]
for m,x,a in cases:
    vals=[]
    for d in (100,200):
        mp.mp.dps=d; vals.append(globals()[m](x,a))
    mp.mp.dps=100
    agree = mp.nstr(abs(vals[0]-vals[1])/abs(vals[1]),3)
    ref=vals[1]; s=getattr(stats.skewnorm,m)(x,a)
    ulp = float(abs(mp.mpf(s)-ref)/abs(ref))/2.22e-16
    print(m,x,a,'scipy',repr(s),'ref',mp.nstr(ref,17),'ulp %.3g'%ulp,'reldiff100v200',agree)
