"""Independent verification of truncnorm findings F1-F5.
Route: cdf/sf/pdf masses by mp.quad of the normal pdf (not erf/ncdf);
quantiles by root finding on an erfc-based cdf, then checked with the quad cdf."""
import mpmath as mp
import numpy as np
from scipy import stats

def phi(t):
    return mp.exp(-t*t/2)/mp.sqrt(2*mp.pi)

def mass_quad(lo, hi):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    if lo == hi:
        return mp.mpf(0)
    if hi <= 0:  # mirror to the right side
        lo, hi = -hi, -lo
    if lo >= 1:
        # far side: phi(lo) * int_0^{hi-lo} exp(-lo*u - u^2/2) du, split on scale 1/lo
        w = hi - lo
        pts = [mp.mpf(0)] + [mp.mpf(k)/lo for k in (1, 4, 16, 64, 256) if mp.mpf(k)/lo < w] + [w]
        return phi(lo)*mp.quad(lambda u: mp.exp(-lo*u - u*u/2), pts)
    pts = [lo]
    for c in (-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40):
        if lo < c < hi:
            pts.append(mp.mpf(c))
    pts.append(hi)
    return mp.quad(phi, pts)

def mass_erf(lo, hi):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    s = mp.sqrt(2)
    if hi <= 0:
        return (mp.erfc(-hi/s) - mp.erfc(-lo/s))/2
    if lo >= 0:
        return (mp.erfc(lo/s) - mp.erfc(hi/s))/2
    return (mp.erf(hi/s) - mp.erf(lo/s))/2

def ref(method, x, a, b):
    a = mp.mpf(a); b = mp.mpf(b); x = mp.mpf(x)
    Z = mass_quad(a, b)
    if method == 'pdf':    return phi(x)/Z
    if method == 'logpdf': return mp.log(phi(x)/Z)
    if method == 'cdf':    return mass_quad(a, x)/Z
    if method == 'logcdf': return mp.log(mass_quad(a, x)/Z)
    if method == 'sf':     return mass_quad(x, b)/Z
    if method == 'logsf':  return mp.log(mass_quad(x, b)/Z)
    q = x
    Ze = mass_erf(a, b)
    if method == 'ppf':
        g = lambda t: mp.log(mass_erf(a, t)/Ze) - mp.log(q)
        inc = True
    else:  # isf
        g = lambda t: mp.log(mass_erf(t, b)/Ze) - mp.log(q)
        inc = False
    lo = a if a != -mp.inf else mp.mpf(-60)
    hi = b if b != mp.inf else mp.mpf(60)
    # bisection on [lo, hi]
    for _ in range(mp.mp.prec + 60):
        m = (lo + hi)/2
        if m == lo or m == hi:
            break
        v = g(m)
        if (v < 0) == inc:
            lo = m
        else:
            hi = m
    xr = (lo + hi)/2
    # check with quad-based cdf
    if method == 'ppf':
        chk = mass_quad(a, xr)/Z
    else:
        chk = mass_quad(xr, b)/Z
    return xr, chk

CASES = [
 ('F1','cdf',1e-20,0,np.inf), ('F1','cdf',1e-12,0,np.inf), ('F1','logcdf',1e-20,0,np.inf),
 ('F1','logsf',1e-20,0,np.inf), ('F1','sf',-1e-300,-np.inf,0), ('F1','sf',-1e-17,-np.inf,0),
 ('F1','logsf',-1e-300,-np.inf,0), ('F1','logcdf',-1e-20,-np.inf,0),
 ('F2','ppf',1e-20,0,np.inf), ('F2','ppf',1e-12,0,np.inf), ('F2','isf',1e-20,-np.inf,0),
 ('F2','ppf',0.4999000000695737,-1e-6,1e-6),
 ('F3','isf',1e-300,30,np.inf), ('F3','isf',1e-20,-1,np.inf), ('F3','isf',1e-10,30,np.inf),
 ('F3','isf',0.9999999999,-50,-10), ('F3','isf',0.99999999999999,-50,-10),
 ('F4','pdf',1.000000005,1,1.00000001), ('F4','logpdf',1.000000005,1,1.00000001),
 ('F4','pdf',0,-1e-6,1e-6),
 ('F5','cdf',0.10000000000000009,0.1,2), ('F5','cdf',-0.0009999999999994458,-0.001,10),
 ('F5','sf',1.0000000089999999,1,1.00000001),
]

def cond_arg(method, x, a, b):
    # cond of cdf/sf wrt x: |x f(x)/F(x)|
    if method in ('cdf', 'sf'):
        F = ref(method, x, a, b)
        f = ref('pdf', x, a, b)
        return abs(mp.mpf(x)*f/F)
    return None

import warnings
warnings.simplefilter('ignore')
for fid, meth, x, a, b in CASES:
    out = []
    for dps in (100, 200):
        mp.mp.dps = dps
        r = ref(meth, x, a, b)
        out.append(r)
    if meth in ('ppf', 'isf'):
        (r1, c1), (r2, _) = out
        r = r2
        qchk = mp.nstr(abs(c1 - x)/x, 3)
    else:
        r1, r2 = out; r = r2; qchk = ''
    agree = mp.nstr(abs(r1 - r2)/abs(r2), 3) if r2 != 0 else 'zero'
    s = getattr(stats.truncnorm, meth)(x, a, b)
    mp.mp.dps = 50
    ulp = abs(mp.mpf(float(s)) - r)/abs(r)/mp.mpf(2.0**-52) if np.isfinite(s) else mp.inf
    c = cond_arg(meth, x, a, b)
    print(f"{fid} {meth}({x!r}, {a}, {b}): scipy={float(s)!r} ref={mp.nstr(r,17)} "
          f"ulp={mp.nstr(ulp,3)} 100v200={agree} qchk={qchk} cond={mp.nstr(c,3) if c is not None else ''}")
