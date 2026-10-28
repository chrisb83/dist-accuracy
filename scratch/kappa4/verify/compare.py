import sys, warnings
sys.path.insert(0, 'C:/Users/chbau/Documents/research/dist-accuracy/scratch/kappa4/verify')
from mpmath import mp
import numpy as np
from scipy.stats import kappa4
import vref

EPS = 2.220446049250313e-16
CASES = [
 # (finding, method, arg, h, k)
 ('F1','cdf',-709.0,-1e6,0.0), ('F1','logcdf',-709.0,-1e6,0.0),
 ('F1','cdf',-6.702276267921237e+31,-1000.0,0.1), ('F1','logcdf',-6.702276267921237e+31,-1000.0,0.1),
 ('F1','cdf',-99.94613906403504,-1e6,-0.01), ('F1','pdf',-99.94613906403504,-1e6,-0.01),
 ('F1','logcdf',-1e300,-1.0,0.0), ('F1','logcdf',-1e300,-0.1,0.1), ('F1','pdf',-1e300,-1000.0,0.1),
 ('F2','ppf',0.36,-1.0,0.0), ('F2','ppf',1e-300,-1.0,0.0), ('F2','ppf',0.36,-1e6,0.0),
 ('F3','ppf',0.9999999999999997,0.1,0.0), ('F3','ppf',0.9999999999999997,-0.1,0.1),
 ('F3','ppf',0.3,1e-08,0.5), ('F3','ppf',0.3678794374926479,1e-08,0.5),
 ('F3','ppf',1e-300,1.0,-0.5), ('F3','ppf',1e-08,1.0,-0.5),
 ('F4','ppf',0.9999999999999999,0.5,1e-08), ('F4','ppf',0.01,0.5,1e-08),
 ('F4','ppf',1e-300,0.5,1e-08), ('F4','ppf',0.3678794374926479,0.0,0.1),
 ('F5','ppf',0.1,-1000.0,0.1), ('F5','ppf',0.9991516571017559,-1e6,-0.01),
 ('F6','cdf',1e-300,1.0,-0.5), ('F6','logcdf',1e-300,1.0,-0.5),
 ('F6','cdf',1e-10,1.0,-0.5), ('F6','cdf',1e-08,1.0,-0.5),
 ('F6','cdf',-2.3025850929940455,0.1,0.0), ('F6','pdf',-0.6931471740943004,0.5,1e-08),
 ('F7','cdf',1.0,0.5,1e-08), ('F7','logcdf',-0.5,0.5,1e-08),
 ('F7','logcdf',2.0,0.5,1e-08), ('F7','cdf',-0.5,0.5,1e-08),
]

def ref(method, arg, h, k):
    return getattr(vref, method)(arg, h, k)

print(f"{'ID':4} {'call':44} {'scipy':26} {'ref(200dps)':26} {'ulp':>10} {'dps-agree'}")
for fid, meth, arg, h, k in CASES:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sv = float(getattr(kappa4, meth)(arg, h, k))
    try:
        mp.dps = 100; r1 = ref(meth, arg, h, k)
        mp.dps = 200; r2 = ref(meth, arg, h, k)
        if r2 == 0:
            agree = 'exact0' if r1 == 0 else 'MISMATCH'
        else:
            d = abs(r1/r2 - 1)
            agree = 'ok' if d < mp.mpf('1e-30') else f'ONLY {mp.nstr(d,2)}'
        mp.dps = 60
        rv = +r2
        if np.isnan(sv) or np.isinf(sv):
            ulp = mp.inf
        else:
            ulp = abs(mp.mpf(sv) - rv)/(abs(rv)*mp.mpf(EPS)) if rv != 0 else mp.inf
        print(f"{fid:4} {meth+'('+repr(arg)+','+repr(h)+','+repr(k)+')':44} {sv!r:26} {mp.nstr(rv,17):26} {mp.nstr(ulp,3):>10} {agree}")
    except Exception as e:
        print(f"{fid:4} {meth+'('+repr(arg)+','+repr(h)+','+repr(k)+')':44} {sv!r:26} REF-FAIL: {e}")
