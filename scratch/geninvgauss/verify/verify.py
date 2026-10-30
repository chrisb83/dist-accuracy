import warnings, mpmath as mp, numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

import functools
@functools.lru_cache(None)
def _logK(p, b, dps):
    return _logK0(p, b)
def logK(p, b):
    return _logK(float(p), float(b), mp.mp.dps)
def _logK0(p, b):
    p = abs(mp.mpf(p)); b = mp.mpf(b)
    # half-integer closed forms (different route than mp.besselk)
    if p == mp.mpf(1)/2:
        return mp.log(mp.sqrt(mp.pi/(2*b))) - b
    if p == mp.mpf(3)/2:
        return mp.log(mp.sqrt(mp.pi/(2*b))*(1+1/b)) - b
    # integral representation K_p(b) = int_0^inf exp(-b cosh t) cosh(p t) dt, scaled by e^b
    f = lambda t: mp.exp(-b*(mp.cosh(t)-1) + p*t)*(1+mp.exp(-2*p*t))/2
    # peak location of exponent: p = b sinh t
    t0 = mp.asinh(p/b)
    w = 1/mp.sqrt(b*mp.cosh(t0)) if b*mp.cosh(t0) > 0 else 1
    pts = [0] + [t0 + k*w for k in (-8,-4,-2,0,2,4,8,16,40) if t0+k*w > 0]
    # finite cutoff T where the integrand is below 10^-(dps+20) relative to the peak
    lpk = -b*(mp.cosh(t0)-1) + p*t0
    T = t0 + 1
    while -b*(mp.cosh(T)-1) + p*T > lpk - (mp.mp.dps+20)*mp.log(10):
        T = t0 + 2*(T - t0)
    pts = sorted(set([q for q in pts if q < T] + [T]))
    return mp.log(mp.quad(f, pts)) - b

def logpdf(x, p, b):
    x, p, b = mp.mpf(x), mp.mpf(p), mp.mpf(b)
    return (p-1)*mp.log(x) - b*(x+1/x)/2 - mp.log(2) - logK(p, b)

def cdf_ig(x, b):  # p = -1/2: IG(mu=1, lambda=b)
    x, b = mp.mpf(x), mp.mpf(b)
    r = mp.sqrt(b/x)
    return mp.ncdf(r*(x-1)) + mp.exp(2*b)*mp.ncdf(-r*(x+1))

def cdf_quad(x, p, b, tail=False):
    # integrate pdf in x (analyst used t=log x); split around mode
    x, p, b = mp.mpf(x), mp.mpf(p), mp.mpf(b)
    lk = logK(p, b)
    g = lambda u: mp.exp((p-1)*mp.log(u) - b*(u+1/u)/2 - mp.log(2) - lk)
    m = ((p-1) + mp.sqrt((p-1)**2 + b*b))/b
    s = 1/mp.sqrt(b) * m
    if not tail:
        pts = sorted(set([mp.mpf(0)] + [v for v in (m-8*s, m-4*s, m-s, m, m+s, m+4*s) if 0 < v < x] + [x/2, x*0.9, x]))
        pts = [v for v in pts if v <= x]
        return mp.quad(g, pts)
    pts = sorted(set([x] + [v for v in (m, m+4*s, m+16*s, 2*x, 10*x) if v > x])) + [mp.inf]
    return mp.quad(g, pts)

cases = [
 ("F1","cdf",1e6,2.3,1.5,"sf"), ("F1","cdf",1e5,2.3,1.5,"sf"), ("F1","cdf",1e10,0.5,1e4,"sf"),
 ("F2","pdf",19800.00005050505,100,0.01,None), ("F2","logpdf",19800.00005050505,100,0.01,None),
 ("F2","logpdf",50,-100,0.01,None), ("F2","cdf",19800,100,0.01,"q"),
 ("F3","logpdf",1.0,0.5,2e9,None), ("F3","pdf",1.0,0.5,2e9,None), ("F3","logpdf",1.0,-0.5,1.2e9,None),
 ("F3","cdf",1.0,0.5,2e9,"refl"),
 ("F4","cdf",0.99,-1.5,1e6,"q"), ("F4","cdf",0.0026758953019778327,-0.5,1,"ig"),
 ("F4","cdf",0.005243131982553904,2.3,1.5,"q"),
 ("F5","cdf",3.2849433241279202,2.3,1.5,"q"), ("F5","cdf",0.6758413057007908,-0.5,1,"ig"),
 ("F6","logpdf",0.9970020029952514,-1.5,1e6,None), ("F6","pdf",0.9970020029952514,-1.5,1e6,None),
 ("F6","pdf",0.99999999,-1.5,1e6,None),
]

def ref(meth, x, p, b, how):
    if meth == "logpdf": return logpdf(x, p, b)
    if meth == "pdf": return mp.exp(logpdf(x, p, b))
    if how == "sf": return 1 - cdf_quad(x, p, b, tail=True)
    if how == "ig": return cdf_ig(x, b)
    if how == "refl":  # cdf(x,p,b) = sf(1/x,-p,b); p=0.5 -> -0.5 IG: sf = 1-cdf_ig
        return 1 - cdf_ig(1/mp.mpf(x), b)
    return cdf_quad(x, p, b)

import sys
rows = []
sel = sys.argv[1:]
for fid, meth, x, p, b, how in cases:
    if sel and fid not in sel: continue
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(ref(meth, x, p, b, how))
    mp.mp.dps = 100
    agree = vals[1] == 0 or abs(vals[0]-vals[1])/abs(vals[1]) < mp.mpf(10)**-30
    r = vals[1]
    s = float(getattr(stats.geninvgauss, meth)(x, p, b))
    ulp = float(abs(mp.mpf(s)-r)/(abs(r)*mp.mpf(2)**-52)) if np.isfinite(s) else float('inf')
    print(f"{fid} {meth}({x!r}, {p}, {b}) scipy={s!r} ref={mp.nstr(r,17)} ulp={ulp:.3g} agree30={agree}")
