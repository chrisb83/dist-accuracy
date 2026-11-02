# Independent reference for tukeylambda (standard form).
# Q(p) = (p**lam - (1-p)**lam)/lam (lam != 0), log(p/(1-p)) (lam == 0).
# cdf: root of Q(exp(t)) = x in t = log p by full-precision bisection,
#      using symmetry F(x) = 1 - F(-x) for x > 0.
# pdf: f(x) = 1/Q'(p) = 1/(p**(lam-1) + (1-p)**(lam-1)) at p = F(-|x|).
# ppf: Q(q) directly.
import mpmath as mp
import numpy as np
from scipy import stats

def Q(p, lam):
    if lam == 0:
        return mp.log(p/(1-p))
    return (p**lam - (1-p)**lam)/lam

def F_low(x, lam):  # x <= 0, returns p = F(x)
    x = mp.mpf(x); lam = mp.mpf(lam)
    if x == 0:
        return mp.mpf(1)/2
    if lam == 0:
        return 1/(1+mp.exp(-x))
    if lam > 0 and x <= -1/lam:
        return mp.mpf(0)
    g = lambda t: Q(mp.exp(t), lam) - x
    lo, hi = mp.mpf(-3000), mp.log(mp.mpf(1)/2)
    # plain bisection to get close, then secant polish
    # full bisection in t = log p (absolute t tolerance -> relative p accuracy)
    for _ in range(int(mp.mp.prec) + 20):
        mid = (lo+hi)/2
        if g(mid) < 0: lo = mid
        else: hi = mid
    return mp.exp((lo+hi)/2)

def cdf(x, lam):
    x = mp.mpf(x)
    return F_low(x, lam) if x <= 0 else 1 - F_low(-x, lam)

def pdf(x, lam):
    x = mp.mpf(x); lam = mp.mpf(lam)
    if lam > 0 and abs(x) >= 1/lam:
        return mp.mpf(0)
    if lam == 0:
        e = mp.exp(-abs(x)); return e/(1+e)**2
    p = F_low(-abs(x), lam)
    return 1/(p**(lam-1) + (1-p)**(lam-1))

def ppf(q, lam):
    return Q(mp.mpf(q), mp.mpf(lam))

REF = dict(cdf=cdf, pdf=pdf, ppf=ppf)
CASES = [
 ('F1','cdf',-1e300,-1), ('F1','cdf',-500,0.001), ('F1','cdf',-7.142857142857071,0.14),
 ('F1','cdf',-1e5,-1), ('F1','pdf',-1e150,-1), ('F1','pdf',1e150,-1),
 ('F1','pdf',-1e230,-3), ('F1','pdf',-500,0.001),
 ('F2','cdf',-10,1e-5), ('F2','cdf',-30,1e-5), ('F2','cdf',10,1e-5), ('F2','pdf',-10,1e-5),
 ('F2','cdf',-3,1.0001e-4),
 ('F3','cdf',-0.4999999,2), ('F3','cdf',-0.019999,50), ('F3','cdf',-6,0.14),
 ('F3','pdf',-6,0.14), ('F3','pdf',-0.0199,50), ('F3','cdf',-0.3,2), ('F3','cdf',-1e-15,-1),
 ('F4','pdf',100,0), ('F4','pdf',700,0), ('F4','pdf',100,1e-5),
 ('F5','ppf',0.49,50), ('F5','ppf',0.42999999999999994,50), ('F5','ppf',0.29,50),
 ('F5','ppf',0.21999999999999997,50), ('F5','ppf',0.5000000001,50),
 ('F5','ppf',0.5000000000000001,3.1321477856738267),
 ('F6','pdf',0.3192697370711285,3.1321477856738267),
 ('F6','cdf',-0.3192697370711285,3.1321477856738267),
]
import warnings; warnings.simplefilter('ignore')
for fid, m, x, lam in CASES:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(REF[m](x, lam))
    mp.mp.dps = 100
    r1, r2 = vals
    agree = (r1 == r2 == 0) or abs(r1 - r2) <= mp.mpf(10)**-30 * abs(r2)
    # check Q(F)=x for cdf
    chk = ''
    if m == 'cdf' and 0 < r2 < 1 and lam != 0:
        mp.mp.dps = 200
        chk = ' Qres=%s' % mp.nstr(abs(Q(r2, mp.mpf(lam)) - mp.mpf(x)) / max(abs(mp.mpf(x)), mp.mpf(1)), 3)
        mp.mp.dps = 100
    s = getattr(stats.tukeylambda, m)(x, lam)
    ref = float(r2)
    if ref == 0:
        ulp = 0.0 if s == 0 else float('inf')
    else:
        ulp = float(abs(mp.mpf(float(s)) - r2) / abs(r2) / mp.mpf(2)**-52)
    print(f"{fid} {m}({x!r}, {lam!r}) scipy={s!r} ref={ref!r} ulp={ulp:.3g} agree30={agree}{chk}")
