import mpmath as mp
import numpy as np
from scipy import stats

# Verifier route: cdf = (1 + a x^-a)^(-1/a) = exp(-log1p(a x^-a)/a)
# pdf = a * x^(-a-1) * (1 + a x^-a)^(-1/a-1); ppf from cdf(x)=q via log-space root,
# plus check cdf(x_ref)==q; quad cross-check of cdf where feasible.
def cdf(x, a):
    x, a = mp.mpf(x), mp.mpf(a)
    return mp.exp(-mp.log1p(a * x**-a) / a)
def sf(x, a):
    x, a = mp.mpf(x), mp.mpf(a)
    return -mp.expm1(-mp.log1p(a * x**-a) / a)
def pdf(x, a):
    x, a = mp.mpf(x), mp.mpf(a)
    return a * x**(-a-1) * mp.exp(-(1/a + 1) * mp.log1p(a * x**-a))
def pdf_doc(t, a):
    return a * (a + t**a)**(-(a+1)/a)
def ppf(q, a):
    q, a = mp.mpf(q), mp.mpf(a)
    # bisection in u = log x; compare log cdf (q<=1/2) or log sf (q>1/2)
    if q <= 0.5:
        g = lambda u: -mp.log1p(a*mp.exp(-a*u))/a - mp.log(q)          # increasing
    else:
        g = lambda u: -(mp.log(-mp.expm1(-mp.log1p(a*mp.exp(-a*u))/a)) - mp.log(1-q))  # increasing
    lo, hi = mp.mpf(-10), mp.mpf(10)
    while g(lo) > 0: lo *= 2
    while g(hi) < 0: hi *= 2
    for _ in range(int(mp.mp.prec) + 60):
        mid = (lo + hi)/2
        if g(mid) > 0: hi = mid
        else: lo = mid
    return mp.exp((lo+hi)/2)
def isf(p, a):
    return ppf(1 - mp.mpf(p), a)   # 1-p exact in mp

cases = [
 ('F1','cdf',3,1000),('F1','cdf',2.1,1000),('F1','cdf',1e160,2),('F1','cdf',1e40,10),
 ('F2','ppf',0.49,1000),('F2','ppf',1e-300,2),('F2','ppf',1e-40,10),('F2','ppf',2.7086801531713753e-16,20),
 ('F3','isf',0.51,1000),('F3','isf',0.9,1000),('F3','isf',0.9999999999999997,20),
 ('F4','ppf',0.9999999999999997,0.1),('F4','ppf',0.9999999999999966,0.1),('F4','ppf',0.99999280314327,0.1),
 ('F4','ppf',0.9999999,0.1),('F4','ppf',0.6,0.001),('F4','ppf',0.5,0.001),('F4','ppf',0.36,0.001),
 ('F5','pdf',5.5,1e-12),('F5','pdf',8.5,1e-12),('F5','cdf',8.5,1e-12),('F5','sf',8.5,1e-12),
 ('F5','cdf',0.5,1e-300),('F5','pdf',0.5,1e-300),('F5','sf',0.5,1e-300),
 ('F5','pdf',8.5,1e-6),('F5','cdf',8.5,1e-6),('F5','sf',8.5,1e-6),
]
fn = dict(cdf=cdf, sf=sf, pdf=pdf, ppf=ppf, isf=isf)
for fid, m, x, a in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(fn[m](x, a))
    mp.mp.dps = 200
    agree = vals[0] == 0 or abs(vals[0]-vals[1]) <= abs(vals[1]) * mp.mpf(10)**-30
    ref = vals[1]
    extra = ''
    if m in ('ppf','isf'):
        q = mp.mpf(x) if m=='ppf' else 1-mp.mpf(x)
        extra = 'cdfchk=%s' % (mp.nstr(abs(cdf(ref,a)/q-1), 3) if q<=0.5 else mp.nstr(abs(sf(ref,a)/(1-q)-1),3))
    sp = float(getattr(stats.kappa3, m)(x, a))
    err = abs(mp.mpf(sp) - ref)/abs(ref)
    ulp = err/mp.mpf(2.220446049250313e-16)
    print(f"{fid} {m}({x!r}, {a!r}) scipy={sp!r} ref={mp.nstr(ref,17)} relerr={mp.nstr(err,3)} ulp={mp.nstr(ulp,3)} agree30={agree} {extra}")

# quad cross-check of cdf route (moderate cases)
mp.mp.dps = 50
for x, a in [(3,1000),(2.1,1000),(8.5,1e-6),(1.5,0.5),(2.0,2.0)]:
    a_ = mp.mpf(a)
    q = mp.quad(lambda t: pdf_doc(t, a_), [0, 1, x]) if x > 1 else mp.quad(lambda t: pdf_doc(t,a_), [0, x])
    print('quad', x, a, mp.nstr(q, 20), mp.nstr(cdf(x, a), 20))
