import warnings; warnings.simplefilter("ignore")
import mpmath as mp
import numpy as np
from scipy import stats
D = stats.genextreme

def t_(x, c):  # (1-cx)^(1/c), or exp(-x) for c=0
    x, c = mp.mpf(x), mp.mpf(c)
    if c == 0: return mp.exp(-x)
    return mp.power(1 - c*x, 1/c)

def pdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    if c == 0: return mp.exp(-mp.exp(-x) - x)
    u = 1 - c*x
    return mp.exp(-mp.power(u, 1/c)) * mp.power(u, 1/c - 1)
def logpdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    if c == 0: return -mp.exp(-x) - x
    u = 1 - c*x
    return -mp.power(u, 1/c) + (1/c - 1)*mp.log(u)
def cdf(x, c): return mp.exp(-t_(x, c))
def sf(x, c): return -mp.expm1(-t_(x, c))
def logcdf(x, c): return -t_(x, c)
def ppf(q, c):  # closed-form inverse, checked by cdf(ppf)==q
    q, c = mp.mpf(q), mp.mpf(c)
    L = -mp.log(q)
    return -mp.log(L) if c == 0 else (1 - mp.power(L, c))/c
def isf(q, c):
    q, c = mp.mpf(q), mp.mpf(c)
    return ppf(1 - q, c)

F = dict(pdf=pdf, logpdf=logpdf, cdf=cdf, sf=sf, logcdf=logcdf, ppf=ppf, isf=isf)
cases = [
 ("F1","cdf",-1e308,1000),("F1","sf",-1e308,1000),("F1","logcdf",-1e308,1000),
 ("F1","logpdf",-1e308,1000),("F1","cdf",1e308,-1000),("F1","sf",1e308,-1000),
 ("F1","logcdf",1e308,-1000),("F1","logpdf",-1.79e308,2),
 ("F2","ppf",0.1290580551878314,1000),("F2","ppf",0.612,-1000),
 ("F2","isf",0.8691479746591456,1000),("F2","isf",0.3863939526808092,-1000),
 ("F3","sf",1.4285714285714286,0.7),("F3","logpdf",1.4285714285714286,0.7),
 ("F3","pdf",1.4285714285714286,0.7),("F3","logpdf",1e12,1e-12),
 ("F4","ppf",0.36787944117144233,0),("F4","ppf",0.36787944117144233,-0.1),
 ("F4","isf",0.6321205588285577,0),
 ("F5","logpdf",1.0,0.99999999),("F5","logpdf",0.9999999800000002,1.00000001),
 ("F5","logpdf",0.9999999998989999,1.000000000001),
 ("F5","logpdf",0.9999999989989999,1.000000000001),
 ("F5","logpdf",0.9999989999989999,1.000000000001),
]
for fid, m, a, c in cases:
    mp.mp.dps = 100; r1 = F[m](a, c)
    mp.mp.dps = 200; r2 = F[m](a, c)
    agree = (r1 == r2 == 0) or abs(r1 - r2) <= abs(r2) * mp.mpf(10)**-30
    extra = ""
    if m in ("ppf", "isf"):
        target = mp.mpf(a) if m == "ppf" else 1 - mp.mpf(a)
        extra = " invchk=%s" % mp.nstr(abs(cdf(r2, c) - target)/target, 3)
    s = float(getattr(D, m)(a, c))
    ref = float(r2)
    err = abs(s - ref)/abs(ref)/2.22e-16 if ref != 0 else float('nan')
    print(f"{fid} {m}({a!r}, {c!r}): scipy={s!r} ref={mp.nstr(r2,17)} ulp={err:.3g} stable={agree}{extra}")
