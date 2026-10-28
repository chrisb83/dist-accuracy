import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
from scipy.stats import gamma
from quadref import P_quad

EPS = 2.220446049250313e-16

def logpdf(x, a):
    a = mp.mpf(a); x = mp.mpf(x)
    return (a-1)*mp.log(x) - x - mp.loggamma(a)

def invert_cdf(a, target, x0, dps=80):
    """Newton on log P(a,x) = log target, P from quadrature."""
    mp.dps = dps
    tgt = mp.log(mp.mpf(target))
    x = mp.mpf(x0)
    for it in range(60):
        P = P_quad(a, x, dps=dps)
        mp.dps = dps
        lp = mp.log(P)
        dlog = logpdf(x, a) - lp          # d log P / dx
        step = (lp - tgt)/dlog
        x = x - step
        if abs(step) < abs(x)*mp.mpf(10)**(-dps+10):
            break
    return x, it

cases = [
    ("isf", 0.9999993371296838, 1e6, 995171.6),
    ("isf", 0.999999938945977,  1e6, 994718.68),
    ("ppf", 1e-08,              1e6, 994398.16),
    ("ppf", 3.4807005884285624e-14, 1e6, 992529.753),
]
for meth, q, a, x0 in cases:
    mp.dps = 80
    target = (1 - mp.mpf(q)) if meth == "isf" else mp.mpf(q)
    for dps in (80, 150):
        xr, it = invert_cdf(a, target, x0, dps=dps)
        if dps == 80:
            xr80 = xr
    mp.dps = 80
    s = getattr(gamma, meth)(q, a)
    u = abs(mp.mpf(s) - xr80)/(abs(xr80)*EPS)
    # conditioning wrt argument q: |dlogx/dlogq|
    lp = mp.log(P_quad(a, xr80, dps=80)); mp.dps = 80
    dlogPdlogx = (logpdf(xr80, a) - lp) * xr80
    cond = abs(1/dlogPdlogx) * (abs(mp.mpf(q))/abs(target) if meth=="isf" else mp.mpf(1))
    print("%s(%.17g, %g): scipy=%.17f ref=%s ulp=%.3g  cond(arg)=%.3g  stab=%.2g  it=%d"
          % (meth, q, a, s, mp.nstr(xr80, 22), float(u), float(cond),
             float(abs(xr80-xr)/abs(xr80)), it))
    # consistency: does scipy's x reproduce the target under the true cdf?
    Ps = P_quad(a, float(s), dps=80); mp.dps = 80
    print("    P(a, x_scipy)=%s  target=%s  rel=%.3g" % (mp.nstr(Ps,18), mp.nstr(target,18), float(abs(Ps-target)/target)))
