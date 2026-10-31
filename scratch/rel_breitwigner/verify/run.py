import sys, warnings
import mpmath as mp
import numpy as np, cmath
from scipy import stats
import vref
warnings.simplefilter('ignore')
cases = [
 ('F1','cdf',1e20,1e160),('F1','cdf',1e300,1e160),('F1','cdf',1e160,1e160),
 ('F2','cdf',5e153,1e154),('F2','cdf',3.5,1e154),('F2','cdf',9.9999999e153,1e154),
 ('F3','cdf',1,1e-160),('F3','cdf',1e-300,1e-160),
 ('F4','pdf',1e-154,1e-154),('F4','pdf',10,1e-154),('F4','pdf',1,1e-160),('F4','pdf',1e-300,1e-160),
 ('F5','pdf',127.6,1e-150),('F5','pdf',1e20,1e-150),('F5','pdf',4.035066294374852e+75,0.001),('F5','pdf',1.2760000000000002e+73,1e-08),
 ('F6','pdf',9.9999999e159,1e160),('F6','pdf',1.00000001e160,1e160),
]
only = sys.argv[1:] 
for fid, m, x, rho in cases:
    if only and fid not in only: continue
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(getattr(vref, m)(x, rho))
    mp.mp.dps = 200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref = vals[1]
    s = getattr(stats.rel_breitwigner, m)(x, rho)
    if ref != 0 and np.isfinite(s):
        ulp = float(abs(mp.mpf(s)-ref)/abs(ref)/mp.mpf(2)**-52)
    else:
        ulp = float('inf')
    print(f"{fid} {m}({x!r}, {rho!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree30={agree}", flush=True)
