# Verifier reference for F1 (johnsonsb pdf): pdf as d/dx of the cdf Phi(a + b*logit(x)),
# computed by mp.diff (different route from the closed-form pdf), plus log-space check.
import mpmath as mp
import numpy as np
from scipy import stats

cases = [(1e-60, 30, 0.5), (3.877170285277934e-60, 30, 0.5), (2.104223637677623e-17, 0, 1)]

def ref_diff(x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    F = lambda t: mp.ncdf(a + b*mp.log(t/(1-t)))
    return mp.diff(F, x, h=x*mp.mpf(10)**(-mp.mp.dps//3))

def ref_log(x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    z = a + b*(mp.log(x) - mp.log1p(-x))
    return mp.exp(mp.log(b) - mp.log(x) - mp.log1p(-x) - z*z/2 - mp.log(2*mp.pi)/2)

for x, a, b in cases:
    vals = {}
    for d in (100, 200):
        mp.mp.dps = d
        vals[d] = (ref_diff(x, a, b), ref_log(x, a, b))
    mp.mp.dps = 200
    agree = [mp.nstr(abs(vals[100][i]/vals[200][i]-1), 3) for i in range(2)]
    cross = mp.nstr(abs(vals[200][0]/vals[200][1]-1), 3)
    r = vals[200][1]
    s = stats.johnsonsb.pdf(x, a, b)
    ulp = float(abs(mp.mpf(s)-r)/(r*mp.mpf(2)**-52)) if s != 0 else float('inf')
    z = a + b*float(np.log(x/(1-x)))
    print(f"pdf({x!r}, {a}, {b}) scipy={s!r} ref={mp.nstr(r,17)} ulp={ulp:.3g} "
          f"z={z:.4f} phi(z)={np.exp(-z*z/2)/np.sqrt(2*np.pi)!r} 100v200={agree} diff-vs-log={cross}")
