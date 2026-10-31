import sys, warnings
import mpmath as mp
import numpy as np
from scipy import stats
warnings.simplefilter('ignore')

# standard form, docstring pdf: f = c phi(x) Phi(-x)^(c-1)
def Phi(x): return mp.erfc(-x/mp.sqrt(2))/2
def _logQ_asym(x):  # x >> 1: asymptotic series of the Mills ratio
    s = mp.mpf(1); t = mp.mpf(1)
    for k in range(1, 12):
        t *= -(2*k-1)/x**2; s += t
    return -x**2/2 - mp.log(x*mp.sqrt(2*mp.pi)) + mp.log(s)
def logQ(x):  # log Phi(-x)
    if x > 1e6: return _logQ_asym(x)
    if x < -1e6: return mp.log1p(-mp.exp(_logQ_asym(-x)))
    if x < 0: return mp.log1p(-mp.erfc(-x/mp.sqrt(2))/2)
    return mp.log(mp.erfc(x/mp.sqrt(2))/2)
def logpdf(x, c):
    return mp.log(c) - x**2/2 - mp.log(mp.sqrt(2*mp.pi)) + (c-1)*logQ(x)
def pdf(x, c):
    return mp.exp(logpdf(x, c))
def pdf_diff(x, c):  # different route: -d/dx sf
    return -mp.diff(lambda t: mp.exp(c*logQ(t)), x)
def cdf_quad(x, c):  # integral of pdf from -inf
    return mp.quad(lambda t: pdf(t, c), [-mp.inf, x - 10, x - 1, x])
def cdf(x, c):
    return -mp.expm1(c*logQ(x))
def logsf(x, c):
    return c*logQ(x)
def quantile_from_logsf(target, c, x0):
    # solve c*logQ(x) = target  (Newton via findroot)
    return mp.findroot(lambda x: c*logQ(x) - target, mp.mpf(x0))
def ppf(q, c, x0):
    return quantile_from_logsf(mp.log1p(-q), c, x0)
def isf(q, c, x0):
    return quantile_from_logsf(mp.log(q), c, x0)

cases = [] if 0 else [
 ('F1','ppf',1e-17,4.445365225459078,-8.67),
 ('F1','ppf',1e-15,4.445365225459078,-8.12),
 ('F1','ppf',2e-16,1e-08,-5.49),
 ('F1','ppf',1e-08,1e-08,0.337),
 ('F1','ppf',1e-08,4.445365225459078,-5.86),
 ('F1','ppf',1e-05,0.001,-2.33),
 ('F2','isf',1e-300,0.5,52.5),
 ('F2','isf',1e-20,0.001,303.5),
 ('F2','isf',1e-300,1e-08,371692.),
 ('F2','ppf',0.9,0.001,67.8),
 ('F2','ppf',0.01,1e-08,1417.8),
 ('F3','isf',0.9999999999522942,1e6,-8.31),
 ('F3','isf',0.9999999994820525,1e6,-8.02),
 ('F3','isf',0.9999993371296838,50,-5.56),
 ('F3','isf',0.99999999999999967,50,-8.54),
 ('F3','ppf',0.01,1e6,-5.61),
 ('F3','ppf',0.5,1e6,-4.83),
 ('F3','ppf',1e-08,1e6,-7.65),
 ('F4','pdf',1000.0,1e-08,None),
 ('F4','pdf',40.0,0.001,None),
 ('F4','pdf',37.6,0.001,None),
 ('F4','pdf',38.0,0.5,None),
 ('F4','pdf',1e5,1e-08,None),
 ('F5','logpdf',1e155,0.001,None),
 ('F5','logsf',1e155,0.001,None),
 ('F5','logpdf',1.4e154,0.5,None),
 ('F5','logpdf',-1.4e154,4.445365225459078,None),
 ('F6','cdf',-38.0,1e10,None),
 ('F6','cdf',-38.5,1e20,None),
 ('F6','logsf',-38.5,1e20,None),
 ('F6','pdf',-38.5,1e20,None),
 ('F7','logpdf',1e5,1e-08,None),
 ('F7','logpdf',1e140,1e-08,None),
 ('F7','logpdf',1e10,1e-08,None),
 ('F7','logpdf',1e100,0.001,None),
 ('F8','pdf',-6.0,1e6,None),
 ('F8','pdf',-5.0,1e6,None),
 ('F8','pdf',-6.0,1e4,None),
]

def ref(meth, a, c, x0):
    a = mp.mpf(a); c = mp.mpf(c)
    if meth == 'ppf':
        x = ppf(a, c, x0); chk = cdf(x, c)/a - 1   # cdf(x_ref) == q
        return x, chk
    if meth == 'isf':
        x = isf(a, c, x0); chk = mp.exp(logsf(x, c))/a - 1
        return x, chk
    if meth == 'pdf':
        v = pdf(a, c)
        alt = pdf_diff(a, c) if abs(a) < 1e4 else v
        return v, alt/v - 1
    if meth == 'logpdf':
        return logpdf(a, c), mp.mpf(0)
    if meth == 'cdf':
        v = cdf(a, c); alt = cdf_quad(a, c)
        return v, alt/v - 1
    if meth == 'logsf':
        v = logsf(a, c)
        if a < 0:   # route via quad of pdf: logsf = log1p(-cdf)
            alt = mp.log1p(-cdf_quad(a, c))
        else:
            alt = v
        return v, alt/v - 1

out = []
for fid, meth, a, c, x0 in cases[int(sys.argv[1]) if len(sys.argv)>1 else 0:]:
    res = {}
    for d in (100, 200):
        mp.mp.dps = d
        res[d] = ref(meth, a, c, x0)
    mp.mp.dps = 100
    v1, v2 = res[100][0], res[200][0]
    agree = abs(v1 - v2) <= abs(v2)*mp.mpf(10)**-30
    s = getattr(stats.powernorm, meth)(a, c)
    if np.isfinite(s) and v2 != 0:
        ulp = float(abs(mp.mpf(float(s)) - v2)/abs(v2)/mp.mpf(2)**-52)
    else:
        ulp = float('inf')
    print(f"{fid}|{meth}({a!r}, {c!r})|{s!r}|{mp.nstr(v2,17)}|{ulp:.3g}|agree30={agree}|selfchk={mp.nstr(res[200][1],3)}")
    sys.stdout.flush()
