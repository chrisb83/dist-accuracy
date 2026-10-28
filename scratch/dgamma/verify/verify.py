# Independent verifier references for dgamma (standard form).
# pdf/logpdf: exp/log of log-space formula with mp.loggamma.
# cdf/sf: P(a,x) via Kummer series x^a e^-x / Gamma(a+1) * 1F1(1; a+1; x)
#         (not mp.gammainc); Q = 1 - P.
# ppf: root of own cdf via mp.findroot (secant), seeded with scipy value.
import sys, time
import mpmath as mp
import numpy as np
from scipy import stats
import warnings; warnings.simplefilter("ignore")

def logpdf(x, a):
    x = mp.mpf(x); a = mp.mpf(a); ax = abs(x)
    return (a-1)*mp.log(ax) - ax - mp.log(2) - mp.loggamma(a)
def pdf(x, a): return mp.exp(logpdf(x, a))
def P(a, x):
    a = mp.mpf(a); x = mp.mpf(x)
    return mp.exp(a*mp.log(x) - x - mp.loggamma(a+1)) * mp.hyp1f1(1, a+1, x)
def cdf(x, a):
    x = mp.mpf(x)
    return mp.mpf(0.5) + P(a, x)/2 if x > 0 else (1 - P(a, -x))/2
def sf(x, a):
    x = mp.mpf(x)
    return (1 - P(a, x))/2 if x > 0 else mp.mpf(0.5) + P(a, -x)/2
def ppf(q, a, x0):
    q = mp.mpf(q)
    return mp.findroot(lambda x: cdf(x, a) - q, mp.mpf(x0), tol=mp.mpf(10)**(-mp.mp.dps+10))
def isf(q, a, x0):
    q = mp.mpf(q)
    return mp.findroot(lambda x: sf(x, a) - q, mp.mpf(x0), tol=mp.mpf(10)**(-mp.mp.dps+10))

F = dict(pdf=pdf, logpdf=logpdf, cdf=cdf, sf=sf)
cases = [
 ("F1","pdf",-150,150),("F1","pdf",120,150),("F1","pdf",1000,110),("F1","pdf",171,171),
 ("F1","pdf",172,172),("F1","pdf",200,200),("F1","pdf",10,200),("F1","pdf",1e6,1e6),
 ("F2","pdf",720,10.5),("F2","pdf",735,10.5),("F2","pdf",744,10.5),("F2","pdf",745,10.5),
 ("F2","pdf",-745,10.5),("F2","pdf",750,10.5),("F2","pdf",720,103),("F2","pdf",750,105),("F2","pdf",800,100),
 ("F3","ppf",0.50000001,1e6),("F3","ppf",0.49999999,1e6),("F3","isf",0.49999999,1e6),("F3","ppf",0.500000005,1e6),
 ("F4","logpdf",995000,1e6),("F4","logpdf",-1003000,1e6),("F4","logpdf",1000000,1e6),("F4","logpdf",1100000,1e6),
 ("F5","logpdf",1.0,1e-310),("F5","logpdf",-1e-300,1e-310),("F5","pdf",1e-300,1e-310),("F5","pdf",1e-100,1e-310),
 ("F7","cdf",995000,1e6),("F7","sf",995000,1e6),("F7","sf",2000,1000),("F7","cdf",-400,200),
]
only = sys.argv[1:] 
for fid, m, x, a in cases:
    if only and fid not in only: continue
    s = float(getattr(stats.dgamma, m)(x, a))
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        t = time.time()
        if m == "ppf": v = ppf(x, a, s); chk = cdf(v, a) - mp.mpf(x)
        elif m == "isf": v = isf(x, a, s); chk = sf(v, a) - mp.mpf(x)
        else: v = F[m](x, a); chk = 0
        vals.append(v)
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    r = vals[1]; rf = float(r)
    if s == rf: ulp = 0.0
    elif not np.isfinite(s) or s == 0: ulp = float('inf')
    else: ulp = float(abs(mp.mpf(s)-r)/(abs(r)*mp.mpf(2)**-52))
    print(f"{fid} {m}({x!r}, {a!r}) scipy={s!r} ref={rf!r} ulp={ulp:.3g} agree30={agree} resid={mp.nstr(chk,3)}", flush=True)
