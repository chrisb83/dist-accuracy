import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
from scipy.stats import gamma
from quadref import P_quad

EPS = 2.220446049250313e-16

def logpdf(x, a):
    a = mp.mpf(a); x = mp.mpf(x)
    return (a-1)*mp.log(x) - x - mp.loggamma(a)

def invert_cdf(a, target, x0, dps=60):
    """Newton on G(x) = log P(a,x) - log target.  G'(x) = pdf/P."""
    mp.dps = dps
    tgt = mp.log(mp.mpf(target))
    x = mp.mpf(x0)
    for it in range(80):
        P = P_quad(a, x, dps=dps)
        mp.dps = dps
        lp = mp.log(P)
        step = (lp - tgt) * mp.e**(lp - logpdf(x, a))   # (G)/(G')
        x = x - step
        if abs(step) < abs(x)*mp.mpf(10)**(-dps+15):
            break
    return x, it

cases = [
    ("isf", 0.9999993371296838, 1e6, 995171.6),
    ("isf", 0.999999938945977,  1e6, 994718.68),
    ("ppf", 1e-08,              1e6, 994398.16),
    ("ppf", 3.4807005884285624e-14, 1e6, 992529.753),
]
for meth, q, a, x0 in cases:
    mp.dps = 60
    target = (1 - mp.mpf(q)) if meth == "isf" else mp.mpf(q)
    xr, it = invert_cdf(a, target, x0, dps=60)
    xr2, it2 = invert_cdf(a, target, x0, dps=110)
    mp.dps = 60
    s = getattr(gamma, meth)(q, a)
    u = abs(mp.mpf(s) - xr)/(abs(xr)*EPS)
    lp = mp.log(P_quad(a, xr, dps=60)); mp.dps = 60
    dlogPdlogx = mp.e**(logpdf(xr, a) - lp) * xr
    if meth == "isf":
        cond = abs(mp.mpf(q)/target) / dlogPdlogx
    else:
        cond = 1/dlogPdlogx
    print("%s(%.17g, %g):" % (meth, q, a))
    print("   scipy = %.17f" % s)
    print("   ref   = %s   (it=%d,%d, dps60 vs 110 rel diff %.2g)"
          % (mp.nstr(xr, 22), it, it2, float(abs(xr-xr2)/abs(xr))))
    print("   ulp = %.3g   dlogP/dlogx = %.4g   cond(arg) = %.3g" % (float(u), float(dlogPdlogx), float(cond)))
    Ps = P_quad(a, float(s), dps=60); mp.dps = 60
    Pr = P_quad(a, xr, dps=60); mp.dps = 60
    print("   P(a,x_scipy)/target - 1 = %.3g ;  P(a,x_ref)/target - 1 = %.3g"
          % (float(Ps/target - 1), float(Pr/target - 1)))
