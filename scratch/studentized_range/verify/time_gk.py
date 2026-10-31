import sys, time; sys.path.insert(0,'.')
import mpmath as mp
from gk_ref import rk, Rk, pdf
mp.mp.dps=100
t=time.time(); print(rk(mp.mpf(1),3), time.time()-t)
t=time.time(); print(Rk(mp.mpf(1),2), mp.erf(mp.mpf(1)/2), time.time()-t)
