import warnings
import mpmath as mp
import numpy as np
from scipy import stats
warnings.simplefilter("ignore")
G = stats.gompertz

def ser_expm1_minus_x(x):
    # e^x - 1 - x via series
    s = mp.mpf(0); t = x
    k = 1
    while True:
        k += 1
        t = t * x / k
        s += t
        if abs(t) < mp.mpf(10)**(-mp.mp.dps-5) * max(abs(s), mp.mpf(10)**-1000):
            break
    return s

def logpdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    if x < 1:
        # log c + (1-c)x - c*(e^x-1-x), series route
        return mp.log(c) + (1 - c)*x - c*ser_expm1_minus_x(x)
    # large x: log c + x - c e^x + c, using e^x = exp(x) directly
    return mp.log(c) + x - c*mp.exp(x) + c

def pdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    if x > 1 and mp.log(c) + x > mp.log(10**6):
        return mp.mpf(0)   # c*e^x > 1e6: pdf < exp(-1e6 + 800), negligible
    return mp.exp(logpdf(x, c))

def cdf_quad(x, c):
    # integral of pdf from 0 to x, split near the mode/end
    x = mp.mpf(x); c = mp.mpf(c)
    pts = [mp.mpf(0)]
    for p in [1, 10, 100, 600, 690, 700, 705, 708, 709, 710, 712, 714, 716, 720]:
        if p < x: pts.append(mp.mpf(p))
    pts.append(x)
    return mp.quad(lambda t: pdf(t, c), pts)

def sf_quad(x, c):
    # integral of pdf from x to infinity (complementary route)
    x = mp.mpf(x); c = mp.mpf(c)
    pts = [x]
    for p in [700, 705, 708, 709, 710, 711, 712, 713, 714, 715, 716, 718, 720, 725, 730, 740]:
        if p > x: pts.append(mp.mpf(p))
    T = mp.log(mp.mpf(10)**6 / c)   # remaining mass is exp(-1e6)
    pts = [p for p in pts if p < T] + [T]
    return mp.quad(lambda t: pdf(t, c), pts)

def cdf_closed(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return -mp.expm1(-c*mp.expm1(x))

def sf_closed(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return mp.exp(-c*mp.expm1(x))

def ppf(q, c):
    q = mp.mpf(q); c = mp.mpf(c)
    return mp.log1p(-mp.log1p(-q)/c)

def isf(p, c):
    p = mp.mpf(p); c = mp.mpf(c)
    return mp.log1p(-mp.log(p)/c)

cases = [
 ("F1","logpdf",1e-20,1, logpdf),
 ("F1","logpdf",1e-150,1, logpdf),
 ("F1","logpdf",1e-12,1, logpdf),
 ("F1","logpdf",1e-8,1, logpdf),
 ("F2","logpdf",710,0.001, logpdf),
 ("F2","logpdf",709.8,0.9474371307510525, logpdf),
 ("F2","logpdf",710,1e-300, logpdf),
 ("F2","logpdf",710,1e-307, logpdf),
 ("F2","pdf",710,1e-307, pdf),
 ("F2","sf",710,1e-307, sf_quad),
 ("F2","sf",713.8013788281542,1e-310, sf_quad),
 ("F2","cdf",710,1e-310, cdf_quad),
 ("F2","cdf",709.79,1e-307, cdf_quad),
 ("F3","ppf",1e-300,1e-310, ppf),
 ("F3","ppf",0.9999999943765867,1e-307, ppf),
 ("F3","isf",3.4807005884285624e-14,1e-307, isf),
 ("F3","isf",1e-300,1e-310, isf),
]

for fid, m, a, c, f in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(f(a, c))
    mp.mp.dps = 100
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref = vals[1]
    extra = ""
    if m in ("ppf", "isf"):
        mp.mp.dps = 100
        # check inversion with cdf / sf (quad and closed form)
        if m == "ppf":
            back = cdf_closed(ref, c); target = mp.mpf(a)
        else:
            back = sf_closed(ref, c); target = mp.mpf(a)
        extra = " inv_relerr=%s" % mp.nstr(abs(back-target)/target, 3)
    if m in ("cdf","sf"):
        mp.mp.dps = 100
        cl = cdf_closed(a,c) if m=="cdf" else sf_closed(a,c)
        extra = " closed_relerr=%s" % mp.nstr(abs(cl-ref)/ref, 3)
    s = getattr(G, m)(a, c)
    rf = float(ref)
    ulp = abs(s - rf)/(abs(rf)*2.220446049250313e-16) if np.isfinite(s) else np.inf
    print(f"{fid} {m}({a!r}, {c!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree30={agree}{extra}")
