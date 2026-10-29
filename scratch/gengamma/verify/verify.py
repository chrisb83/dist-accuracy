# Independent verification of gengamma F1, F2, F3, F5.
# cdf route: P(a, y) = 1/Gamma(a+1) * int_0^{y^a} exp(-u^(1/a)) du  (u = t^a substitution),
# with y^a = x^(c a) formed in log space. Quantiles: findroot in log y on this P, then
# checked against the series P = y^a e^-y sum y^n / Gamma(a+n+1).
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")

def P_quad(a, logY):          # logY = log(y^a)
    # scaled to [0, 1] so mp.quad's absolute tolerance is relative to the value
    Y = mp.exp(logY)
    f = lambda v: mp.exp(-(Y*v)**(1/a))
    pts = [0, 1] if Y <= mp.mpf('0.5') else [0, mp.mpf('0.5')/Y, 1/Y, 1] if Y > 1 else [0, mp.mpf('0.5')/Y, 1]
    return Y * mp.quad(f, pts) / mp.gamma(a + 1)

def P_series(a, logy):
    y = mp.exp(logy)
    s = mp.nsum(lambda n: y**n / mp.gamma(a + n + 1), [0, mp.inf])
    return mp.exp(a*logy - y) * s

def cdf(x, a, c):
    a, c, x = mp.mpf(a), mp.mpf(c), mp.mpf(x)
    P = P_quad(a, c*a*mp.log(x))
    return P if c > 0 else 1 - P
def sf(x, a, c):
    a, c = mp.mpf(a), mp.mpf(c)
    return 1 - cdf(x, a, c)

def ppf(q, a, c, lower=True):
    a, c, q = mp.mpf(a), mp.mpf(c), mp.mpf(q)
    # target for P(a, y): ppf c>0 and isf c<0 -> q; otherwise 1 - q
    p = q if (c > 0) == lower else 1 - q
    # solve log P_series(a, y) = log p in L = log y; start from small-y asymptote
    L0 = (mp.log(p) + mp.loggamma(a + 1)) / a
    if L0 > -1:
        L0 = mp.log(mp.mpf(2))
    L = mp.findroot(lambda L: mp.log(P_series(a, L)) - mp.log(p), L0)
    return mp.exp(L / c)
def isf(q, a, c):
    return ppf(q, a, c, lower=False)

def logpdf(x, a, c):
    a, c, x = mp.mpf(a), mp.mpf(c), mp.mpf(x)
    return mp.log(abs(c)) + (c*a - 1)*mp.log(x) - x**c - mp.loggamma(a)
def pdf(x, a, c):
    return mp.exp(logpdf(x, a, c))

F = dict(cdf=cdf, sf=sf, ppf=ppf, isf=isf, logpdf=logpdf, pdf=pdf)
third = 1/3
cases = [
 ("F1","cdf",1e-40,0.001,10), ("F1","cdf",1e-300,0.001,10), ("F1","cdf",1e-200,third,3),
 ("F1","sf",1e40,0.001,-10), ("F1","sf",1e-40,0.001,10), ("F1","cdf",1e40,0.001,-10),
 ("F1","cdf",4.416325735416545e-33,0.001,10), ("F1","cdf",1e-105,third,3),
 ("F1","cdf",1.4132001038995107e-108,third,3),
 ("F2","ppf",0.1,0.001,10), ("F2","isf",0.9,0.001,10), ("F2","ppf",0.9,0.001,-10),
 ("F2","isf",0.1,0.001,-10), ("F2","ppf",1e-150,third,3), ("F2","isf",1e-150,third,-3),
 ("F3","logpdf",31.118759991335068,1000,2), ("F3","pdf",33.166247903554,1000,2),
 ("F5","ppf",0.57,2,0.001), ("F5","isf",0.43,2,0.001),
]
if __name__ == '__main__':
 for fid, m, x, a, c in cases:
     vals = []
     for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(F[m](x, a, c))
     mp.mp.dps = 200
     agree = abs(vals[0] - vals[1]) <= abs(vals[1]) * mp.mpf(10)**-30
     ref = vals[1]
     extra = ""
     if m in ("ppf", "isf"):   # check cdf(x_ref) == q via the quad route
        qq = cdf(ref, a, c) if m == "ppf" else sf(ref, a, c)
        extra = " invchk=%s" % mp.nstr(abs(qq - x) / x, 3)
     s = getattr(stats.gengamma, m)(x, a, c)
     r = float(ref)
     ulp = abs(s - r) / (abs(r) * 2.220446049250313e-16) if np.isfinite(s) else np.inf
     print(f"{fid} {m}({x!r}, {a!r}, {c!r}) scipy={s!r} ref={mp.nstr(ref, 17)} ulp={ulp:.3g} agree30={agree}{extra}")
