import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter('ignore')
ew = stats.exponweib

# dF/dv in v = log(s), s = x**c:  a (1-e^-s)^(a-1) e^-s s
def g(v, a):
    s = mp.exp(v)
    return a * (-mp.expm1(-s))**(a-1) * mp.exp(-s) * s

def cdf_q(x, a, c):
    t = x**c
    if t < 1:
        # substitute w = s**a: F = int_0^{t^a} ((1-e^-s)/s)^(a-1) e^-s dw, integrand ~ 1
        def h(w):
            s = w**(1/a)
            return (-mp.expm1(-s)/s)**(a-1) * mp.exp(-s) if s > 0 else mp.mpf(1)
        return mp.quad(h, [0, t**a])
    # t >= 1: independent identity exp(a*log1p(-exp(-t)))
    return mp.exp(a*mp.log1p(-mp.exp(-t)))

def sf_q(x, a, c):
    v0 = c*mp.log(x)
    stop = mp.exp(v0) + mp.log(a + 1) + (mp.mp.dps + 20)*mp.log(10) + 10
    vtop = mp.log(stop)
    if v0 >= vtop: raise ValueError
    n = 40
    pts = [v0 + (vtop - v0)*mp.mpf(k)/n for k in range(n+1)]
    return mp.quad(lambda v: g(v, a), pts)

def logpdf_m(x, a, c):
    t = x**c
    return mp.log(a)+mp.log(c)+(a-1)*mp.log(-mp.expm1(-t)) - t + (c-1)*mp.log(x)

def ppf_m(q, a, c):  # closed form, then checked by quad cdf
    return (-mp.log1p(-q**(1/a)))**(1/c)

def isf_m(p, a, c):
    return ppf_m(1 - p, a, c)

cases_all = [
 ('F1','sf',1e-30,0.01,1), ('F1','sf',9.99999999999999e-18,0.01,1), ('F1','sf',1e-12,0.01,1),
 ('F1','sf',0.457088189614875,0.3,50),
 ('F2','isf',0.5,0.01,1), ('F2','isf',0.9,0.01,1), ('F2','isf',0.29,0.01,1),
 ('F2','isf',0.999999938945977,0.3,50),
 ('F3','ppf',0.9999999999522942,1e6,1), ('F3','ppf',0.9999999999999627,1000,0.5),
 ('F3','ppf',0.99,1e6,1), ('F3','ppf',0.5,1e6,1), ('F3','ppf',0.9999999999999997,2,0.01),
 ('F4','ppf',1e-100,0.3,50), ('F4','ppf',1e-300,0.3,50),
 ('F5','logpdf',1e-200,2.8923945291034436,1.9505288745913174), ('F5','logpdf',1e-100,0.3,50),
 ('F5','pdf',1e-20,0.3,50), ('F5','cdf',1e-20,0.3,50),
 ('F6','cdf',9.5,1e6,1), ('F6','logpdf',9.5,1e6,1), ('F6','pdf',9.5,1e6,1),
 ('F7','sf',720.0000000000001,1e6,1),
]

cases = [k for k in cases_all if k[0] in ('F4','F7') or (k[0] in ('F5','F6') and k[1]=='cdf') or k[2] in (0.5,0.29)] + [('F2','isf',0.9,0.01,1)]

def ref(meth, x, a, c):
    x, a, c = mp.mpf(x), mp.mpf(a), mp.mpf(c)
    if meth == 'sf': return sf_q(x, a, c), None
    if meth == 'cdf': return cdf_q(x, a, c), None
    if meth == 'logpdf': return logpdf_m(x, a, c), None
    if meth == 'pdf': return mp.exp(logpdf_m(x, a, c)), None
    if meth == 'ppf':
        r = ppf_m(x, a, c); chk = cdf_q(r, a, c) if x < 0.5 else 1 - sf_q(r, a, c)
        return r, (chk - x)/x
    if meth == 'isf':
        r = isf_m(x, a, c); chk = sf_q(r, a, c) if x < 0.5 else 1 - cdf_q(r, a, c)
        return r, (chk - x)/x

import time
for fid, meth, x, a, c in cases:
    t0 = time.time()
    mp.mp.dps = 100; r1, k1 = ref(meth, x, a, c)
    mp.mp.dps = 200; r2, k2 = ref(meth, x, a, c)
    agree = mp.nstr(abs(r1-r2)/abs(r2), 3)
    s = float(getattr(ew, meth)(x, a, c))
    rf = float(r2)
    ulp = abs(s - rf)/(abs(rf)*2.220446049250313e-16) if np.isfinite(s) else float('inf')
    print(f"{fid} {meth}({x!r}, {a}, {c}) scipy={s!r} ref={mp.nstr(r2,17)} ulp={ulp:.3g} 100v200={agree} invchk={mp.nstr(k2,3) if k2 is not None else '-'} t={time.time()-t0:.1f}s", flush=True)
