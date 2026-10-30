"""Independent verifier reference for invgauss (standard form, loc=0, scale=1).
pdf from docstring: f = (2 pi x^3)^(-1/2) exp(-(x-mu)^2/(2 mu^2 x)).
cdf route (differs from closed form Phi(z1)+exp(2/mu)Phi(-z2)):
  term2 = exp(2/mu) Phi(-z2) = exp(-z1^2/2) * M(z2), M(z) = U(1/2,1/2,z^2/2)/(2 sqrt(pi))
  (uses 2/mu - z2^2/2 = -z1^2/2 exactly and erfc(u) exp(u^2) = U(1/2,1/2,u^2)/sqrt(pi)).
  sf = exp(-z1^2/2) (M(z1) - M(z2)) for z1 > 0, else 1 - cdf.
Cross-check for moderate points: mp.quad of the pdf.
Quantiles: geometric bisection on log cdf / log sf.
"""
import sys, warnings
import mpmath as mp
import numpy as np
from scipy import stats

def M(z):  # erfc(z/sqrt2) exp(z^2/2) / 2, z > 0
    return mp.hyperu(mp.mpf(1)/2, mp.mpf(1)/2, z*z/2) / (2*mp.sqrt(mp.pi))

def zs(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    s = mp.sqrt(x)
    return (x/mu - 1)/s, (x/mu + 1)/s

def cdf(x, mu):
    z1, z2 = zs(x, mu)
    t2 = mp.exp(-z1*z1/2) * M(z2)
    return mp.ncdf(z1) + t2

def sf(x, mu):
    z1, z2 = zs(x, mu)
    if z1 > 0:
        return mp.exp(-z1*z1/2) * (M(z1) - M(z2))
    return 1 - cdf(x, mu)

def logpdf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    return -mp.log(2*mp.pi*x**3)/2 - (x - mu)**2/(2*mu**2*x)

def pdf(x, mu):
    return mp.exp(logpdf(x, mu))

def sf_quad(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    sc = 2*mu**2  # tail scale
    pts = [x + k*sc*min(1, x/sc + 1) for k in (0, 0.01, 0.1, 1, 10, 100)] + [mp.inf]
    pts = sorted(set(pts))
    return mp.quad(lambda t: pdf(t, mu), pts)

def quantile(q, mu, upper=False):
    q = mp.mpf(q)
    lq = mp.log(q)
    f = (lambda x: mp.log(sf(x, mu))) if upper else (lambda x: mp.log(cdf(x, mu)))
    lo, hi = mp.mpf(-1000), mp.mpf(1000)  # log x
    for _ in range(mp.mp.prec + 40):
        mid = (lo + hi)/2
        v = f(mp.exp(mid))
        if (v < lq) != upper:
            lo = mid
        else:
            hi = mid
    return mp.exp((lo + hi)/2)

meth = {
 'pdf': pdf, 'logpdf': logpdf, 'cdf': cdf, 'sf': sf,
 'logcdf': lambda x, m: mp.log(cdf(x, m)), 'logsf': lambda x, m: mp.log(sf(x, m)),
 'ppf': lambda q, m: quantile(q, m), 'isf': lambda q, m: quantile(q, m, True),
}

CALLS = [
 ('F1','sf',1e40,1e20), ('F1','logsf',1e40,1e20), ('F1','logcdf',1e40,1e20),
 ('F1','sf',1e40,1e60), ('F1','logsf',1e30,1e20), ('F1','sf',1e19,1e8),
 ('F1','sf',1e9,1e3), ('F1','logcdf',1e9,1e3), ('F1','sf',1e3,1.0),
 ('F2','isf',1e-100,1.0), ('F2','isf',1e-300,1.0), ('F2','isf',1e-300,0.01),
 ('F2','isf',1.778279410038923e-76,1000.0), ('F2','isf',1e-20,1e20),
 ('F3','ppf',1e-300,0.14546264555347513), ('F3','ppf',1.21e-26,0.14546264555347513),
 ('F4','isf',1e-17,1e-3),
 ('F5','ppf',0.01,1e-20), ('F5','ppf',0.57,1e-8), ('F5','isf',1e-16,0.001),
 ('F6','logpdf',1e300,1e100), ('F6','logpdf',1.4e154,1.0),
 ('F7','pdf',1e103,1e60),
 ('F8','cdf',1e-308,1e-308), ('F8','cdf',2e-310,1e-310),
 ('F9','logcdf',1e-23,1e-20),
 ('F10','cdf',1.00000001e-8,1e-8), ('F10','sf',0.002,0.001),
]
QUAD = [('sf',1e9,1e3), ('sf',1e3,1.0), ('sf',1e19,1e8), ('sf', 0.002, 0.001)]

def rel(a, b):
    return abs(a - b)/abs(b) if b != 0 else abs(a - b)

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    for fid, m, a, s in CALLS:
        vals = []
        for d in (100, 200):
            mp.mp.dps = d
            vals.append(meth[m](a, s))
        mp.mp.dps = 100
        agree = rel(vals[0], vals[1]) < mp.mpf('1e-30')
        ref = vals[1]
        try:
            with np.errstate(all='ignore'):
                sp = float(getattr(stats.invgauss, m)(a, s))
        except Exception as e:
            sp = f'{type(e).__name__}'
        if isinstance(sp, float) and np.isfinite(sp) and ref != 0:
            ulp = float(rel(mp.mpf(sp), ref))/2.220446049250313e-16
        else:
            ulp = float('inf')
        extra = ''
        if m in ('ppf', 'isf'):
            mp.mp.dps = 100
            chk = (sf if m == 'isf' else cdf)(ref, s)
            extra = f' check={mp.nstr(chk, 8)}'
        print(f'{fid} {m}({a!r}, {s!r}) scipy={sp!r} ref={mp.nstr(ref, 17)} ulp={ulp:.3g} agree30={agree}{extra}')
    print('--- quad cross-check')
    for m, a, s in QUAD:
        mp.mp.dps = 50
        v1 = sf(a, s); v2 = sf_quad(a, s)
        print(m, a, s, mp.nstr(v1, 20), mp.nstr(v2, 20), mp.nstr(rel(v1, v2), 3))
