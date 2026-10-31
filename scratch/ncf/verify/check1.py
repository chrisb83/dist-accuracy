import sys, warnings; sys.path.insert(0, '.')
import mpmath as mp, numpy as np
from scipy import stats
from vref import pdf, cdf, sf, both
warnings.simplefilter('ignore')
nc27 = 0.41578441799226107
cases = [
 ('F1','sf',1,27,27,0),
 ('F2','cdf',1e-210,1,1,1),
 ('F3','cdf',10,1,1,1e-20),
 ('F3','sf',10,1,1,1e-300),
 ('F4','pdf',1e16,27,27,nc27),
 ('F4','pdf',1e8,27,27,nc27),
 ('F7','sf',55,100,1000,50),
 ('F7','sf',50,100,1000,50),
 ('F8','sf',1e308,1,0.01,1),
 ('F8','cdf',1e308,1,0.01,1),
]
fns = dict(pdf=pdf, cdf=cdf, sf=sf)
for fid, meth, x, *s in cases:
    try: sv = getattr(stats.ncf, meth)(x, *s)
    except Exception as e: sv = repr(e)
    r, ok = both(fns[meth], x, *s)
    rf = float(r)
    err = abs(sv - rf)/abs(rf)/2.22e-16 if isinstance(sv, float) and np.isfinite(sv) else float('inf')
    print(fid, meth, x, s, 'scipy', repr(sv), 'ref', mp.nstr(r, 17), 'stable', ok, 'ulp %.3g' % err, flush=True)
