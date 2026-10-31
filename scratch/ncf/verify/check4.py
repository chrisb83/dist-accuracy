import sys, warnings; sys.path.insert(0, '.')
import mpmath as mp, numpy as np
from scipy import stats
from vref import pdf, cdf, sf
warnings.simplefilter('ignore')
nc27 = 0.41578441799226107
cases = [ # id, method, q, shapes, analyst x
 ('F2','ppf',1e-110,(1,1,1),6.707091574390156e-220),
 ('F5','isf',1e-213,(27,27,nc27),2.011679359431735e+16),
 ('F5','isf',1e-10,(1,1,1),8.664571782420834e+19),
 ('F6','ppf',0.99,(5,0.5,2),44898971.722370245),
 ('F6','ppf',0.9999999943765867,(5,0.5,2),4.489897079322275e+32),
]
def solve(meth, q, s, x0, dps):
    with mp.workdps(dps):
        q = mp.mpf(q); x = mp.mpf(x0)
        if meth == 'ppf' and q > 0.5: meth, q = 'isf', 1 - q
        for _ in range(3):
            if meth == 'ppf': x = x - (cdf(x,*s) - q)/pdf(x,*s)
            else: x = x + (sf(x,*s) - q)/pdf(x,*s)
        res = (cdf(x,*s) - q)/q if meth=='ppf' else (sf(x,*s)-q)/q
        return x, res
for fid, meth, q, s, xa in cases:
    try: sv = float(getattr(stats.ncf, meth)(q, *s))
    except Exception as e: sv = type(e).__name__
    x1, r1 = solve(meth, q, s, xa, 100); x2, r2 = solve(meth, q, s, xa, 200)
    stable = abs((x1-x2)/x2) < mp.mpf(10)**-30
    rel_an = float(abs((mp.mpf(xa)-x2)/x2))
    ulp = abs(sv - float(x2))/float(x2)/2.22e-16 if isinstance(sv, float) else None
    print(fid, meth, q, s, 'scipy', sv, 'mine', mp.nstr(x2, 17), 'resid %.1e' % float(r2), 'stable', stable, 'analyst rel diff %.1e' % rel_an, 'ulp', ulp, flush=True)
