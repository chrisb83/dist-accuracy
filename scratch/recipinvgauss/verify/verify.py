# Independent verifier reference: cdf/sf by mp.quad of the docstring pdf,
# logpdf by the rewritten form -(1/mu - x)^2/(2x) - log(2 pi x)/2.
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")

def pdf(t, mu):
    return mp.exp(-(1/mu - t)**2/(2*t)) / mp.sqrt(2*mp.pi*t)

def logpdf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    return -(1/mu - x)**2/(2*x) - mp.log(2*mp.pi*x)/2

def breakpoints(a, b, mu):
    # scale c = 1/|d logpdf/dt| at the interval end that holds the mass,
    # geometric steps away from it; plus mode-region points; plus decades.
    m = 1/mu; s = mp.sqrt(m)
    def scale(t):
        d = -(1 - 1/(mu*t)**2)/2 - 1/(2*t)
        return 1/abs(d) if d != 0 else s
    pts = {a, b}
    for k in range(-40, 41):
        p = m + k*s
        if a < p < b: pts.add(p)
    if mp.isinf(b):
        c = scale(a)
        for k in range(0, 60):
            pts.add(a + c*mp.mpf(2)**k)
    else:
        c = scale(b)
        for k in range(0, 60):
            p = b - c*mp.mpf(2)**k
            if a < p < b: pts.add(p)
        for k in range(1, 31):
            pts.add(b*mp.mpf(10)**(-k))
    return sorted(pts)

def cdf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    return mp.quad(lambda t: pdf(t, mu), breakpoints(mp.mpf(0), x, mu))

def sf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    return mp.quad(lambda t: pdf(t, mu), breakpoints(x, mp.inf, mu))

REF = {'cdf': cdf, 'sf': sf, 'logpdf': logpdf,
       'pdf': lambda x, mu: mp.exp(logpdf(x, mu))}

calls = [
 ('F1','cdf',1000,0.001),('F1','cdf',500,0.001),('F1','sf',1000,0.001),
 ('F1','sf',1e-10,0.001),('F1','sf_trivial',1e300,1e-200),
 ('F2','cdf',1e-40,1e160),('F2','cdf',1e-30,1e160),('F2','cdf',1e-19,1e8),
 ('F2','cdf',1e-14,1e8),('F2','cdf',1e-9,1000),
 ('F3','cdf',0.07142857142857142,0.1),('F3','cdf',6.5,0.01),
 ('F3','cdf',79.36507936507935,0.003),('F3','cdf',166.66666666666666,0.003),
 ('F3','sf',1400,0.1),('F3','sf',1400,0.01),('F3','sf',666.6666666666666,0.003),
 ('F4','logpdf',1,1.5e154),('F4','pdf',1,1e160),('F4','logpdf',5e-324,1e160),
 ('F4','pdf',1e-321,1e160),('F4','logpdf',1.3e-6,1e160),
 ('F5','logpdf',1.4e154,1),('F5','logpdf',2.2220716924065444e+154,0.6300426780936912),
 ('F5','logpdf',1e300,1e8),('F5','logpdf',1e300,1e-100),
 ('F6','logpdf',2e162,1e-162),('F6','logpdf',2e200,1e-200),('F6','logpdf',1e200,1e-200),
 ('F6','logpdf',2e160,1e-160),
]
for fid, meth, x, mu in calls:
    if meth == 'sf_trivial':
        # cdf(1e300, 1e-200) = 1 - sf(1e300); sf is exp(-~5e299): cdf == 1 exactly in double
        meth2 = 'cdf'; vals = []
        for d in (100, 200):
            mp.mp.dps = d; vals.append(1 - sf(x, mu))
    else:
        meth2 = meth; vals = []
        for d in (100, 200):
            mp.mp.dps = d; vals.append(REF[meth](x, mu))
    mp.mp.dps = 200
    agree = (vals[0] == vals[1]) or abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    r = vals[1]
    s = getattr(stats.recipinvgauss, meth2)(x, mu)
    if np.isfinite(s) and r != 0:
        ulp = float(abs(mp.mpf(float(s)) - r)/abs(r)/mp.mpf(2.220446049250313e-16))
    else:
        ulp = float('inf')
    print(f"{fid} {meth2}({x!r}, {mu!r}) scipy={s!r} ref={mp.nstr(r,17)} ulp={ulp:.3g} agree30={agree}", flush=True)
