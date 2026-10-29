import sys
import mpmath as mp
import numpy as np
from scipy import stats

def logpdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return (mp.log(x+1) - mp.log(2*c) - mp.log(2*mp.pi)/2 - 3*mp.log(x)/2
            - (x-1)**2/(2*x*c**2))

def pdf(x, c):
    return mp.exp(logpdf(x, c))

def z_of(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return (x-1)/(c*mp.sqrt(x))   # algebraically equal to (sqrt x - 1/sqrt x)/c

def sf_quad(x, c):
    # direct integration of the pdf over [x, inf), substituting x = 1 + c*u*sqrt(...)
    x = mp.mpf(x); c = mp.mpf(c)
    pts = [x] + [x + c*k for k in (1, 3, 10, 30)] + [mp.inf]
    return mp.quad(lambda t: pdf(t, c), pts)

def sf_erfc(x, c):
    return mp.erfc(z_of(x, c)/mp.sqrt(2))/2

def cdf_erfc(x, c):
    return mp.erfc(-z_of(x, c)/mp.sqrt(2))/2

def norm_ppf(q):
    q = mp.mpf(q)
    # solve Phi(z) = q in log space
    if q < 0.5:
        f = lambda z: mp.log(mp.erfc(-z/mp.sqrt(2))/2) - mp.log(q)
        z0 = -mp.sqrt(-2*mp.log(q)) if q < 1e-3 else mp.mpf(-1)
    else:
        f = lambda z: mp.log(mp.erfc(z/mp.sqrt(2))/2) - mp.log(1-q)
        z0 = mp.mpf(1)
    return mp.findroot(f, z0)

def ppf(q, c):
    t = mp.mpf(c)*norm_ppf(q)
    # conjugate form, no cancellation for t < 0
    return 4/(mp.sqrt(t**2+4) - t)**2

def isf(q, c):
    return ppf(1 - mp.mpf(q), c)

cases = [
 ('F1','ppf',0.4,1e100),('F1','ppf',0.1,1e9),('F1','isf',0.9,1e100),
 ('F1','ppf',0.1,1e8),('F1','ppf',0.1,1e6),('F1','isf',0.9,1e6),
 ('F1','ppf',0.1,1000.),('F1','ppf',1e-300,29.),('F1','ppf',0.01,29.),
 ('F2','logpdf',1.4e154,29.),('F2','logpdf',1e300,1e100),
 ('F2','pdf',1e160,1e100),('F2','pdf',1.4e154,1e100),
 ('F3','logpdf',1.0,1e-170),('F3','pdf',1.0,1e-170),
 ('F4','sf',1.0000000000000002,1e-17),('F4','cdf',1.0000000000000002,1e-155),
 ('F4','cdf',1.0000000000000002,1e-17),
]

def ref(meth, a, c):
    if meth == 'ppf': return ppf(a, c)
    if meth == 'isf': return isf(a, c)
    if meth == 'logpdf': return logpdf(a, c)
    if meth == 'pdf': return pdf(a, c)
    if meth == 'sf': return sf_quad(a, c)
    if meth == 'cdf':
        # cdf = 1 - sf (sf by quadrature)
        return 1 - sf_quad(a, c)

def run(dps):
    mp.mp.dps = dps
    return [ref(m, a, c) for (_, m, a, c) in cases]

r1 = run(100); r2 = run(200)
mp.mp.dps = 50
import warnings; warnings.simplefilter('ignore')
for (fid, m, a, c), v1, v2 in zip(cases, r1, r2):
    agree = (v1 == v2) or abs(v1-v2) <= abs(v2)*mp.mpf(10)**-30
    s = getattr(stats.fatiguelife, m)(a, c)
    extra = ''
    mp.mp.dps = 200
    if m in ('ppf','isf'):
        x = v2
        q = cdf_erfc(x, c) if m == 'ppf' else sf_erfc(x, c)
        extra = 'inv_relerr=%s' % mp.nstr(abs(q - a)/a, 3)
    if m in ('sf','cdf'):
        alt = sf_erfc(a, c) if m == 'sf' else cdf_erfc(a, c)
        extra = 'erfc_relerr=%s' % mp.nstr(abs(alt - v2)/abs(v2), 3)
    mp.mp.dps = 50
    ref_f = float(v2)
    ulp = abs(mp.mpf(s) - v2)/(abs(v2)*mp.mpf(2)**-52) if np.isfinite(s) else mp.inf
    print(f"{fid} {m}({a!r}, {c!r}) scipy={s!r} ref={ref_f!r} ulp={mp.nstr(ulp,3)} agree30={agree} {extra}")
