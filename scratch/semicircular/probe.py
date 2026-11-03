import sys; sys.path.insert(0, 'refs'); sys.path.insert(0, '.claude/skills/dist-accuracy-review/scripts')
import numpy as np, mpmath as mp, scipy.special as sc
from scipy import stats
import semicircular_ref as R
mp.mp.dps = 100
S = stats.semicircular
def ulp(v, r): r = mp.mpf(r); return float(abs(mp.mpf(v) - r) / abs(r) / 2.220446049250313e-16) if np.isfinite(v) else float('inf')
print("cdf terms at x=-1+d: abs errors of x*sqrt(1-x*x) and asin(x), total abs err")
for x in [-0.9999, -0.99999999, 0.99999999, -0.999999999999]:
    X = mp.mpf(x)
    t1 = x*np.sqrt(1-x*x); t1r = X*mp.sqrt(1-X*X)
    t1b = x*np.sqrt((1-x)*(1+x))
    a = np.arcsin(x); ar = mp.asin(X)
    tot = S.cdf(x) - R.cdf(X)
    alt = 0.5+1.0/np.pi*(x*np.sqrt((1-x)*(1+x)) + np.arcsin(x))
    bet = sc.betainc(1.5, 1.5, (1+x)/2) if x < 0 else sc.betaincc(1.5, 1.5, (1-x)/2)
    print(x, "t1 err %.2e (with (1-x)(1+x): %.2e)  asin err %.2e  total %.2e  ref %.3e  alt(1-x)(1+x) ulp %.3g  betainc ulp %.3g" % (
        float(t1 - t1r), float(t1b - t1r), float(a - ar), float(tot), float(R.cdf(X)), ulp(alt, R.cdf(X)), ulp(bet, R.cdf(X))))
print("pdf with (1-x)(1+x):")
for x in [1-2**-53, -1+2**-53*2, 1-1e-8, -1+1e-4]:
    X = mp.mpf(x); r = R.pdf(X)
    print(x, "scipy ulp %.3g  alt ulp %.3g" % (ulp(S.pdf(x), r), ulp(2/np.pi*np.sqrt((1-x)*(1+x)), r)))
print("inherited demos")
for x in [1-1e-8, 1-1e-12]:
    X = mp.mpf(x); r = R.sf(X); print("sf", x, S.sf(x), mp.nstr(r, 16), "%.3g" % ulp(S.sf(x), r), " betaincc((1+x)/2... ) via cdf(-x):", "%.3g" % ulp(sc.betainc(1.5,1.5,(1-x)/2), r))
x = -1+1e-12; X = mp.mpf(x); r = mp.log(R.cdf(X)); print("logcdf", x, S.logcdf(x), mp.nstr(r, 16), "%.3g" % ulp(S.logcdf(x), r))
x = 1-1e-12; X = mp.mpf(x); r = mp.log(R.sf(X)); print("logsf", x, S.logsf(x), mp.nstr(r, 16), "%.3g" % ulp(S.logsf(x), r))
mp.mp.dps = 400
for q in [1e-20, 1e-10]:
    r = R.isf(mp.mpf(q)); print("isf", q, repr(S.isf(q)), mp.nstr(r, 20), "%.3g" % ulp(S.isf(q), r), " -ppf(q):", "%.3g" % ulp(-S.ppf(q), r))
print("ppf near 0.5 via beta_ppf p:")
from scipy.special import _ufuncs as scu
import scipy.stats._continuous_distns as cd
for q in [0.49999999999999, 0.4999999999999999]:
    p = scu._beta_ppf(q, 1.5, 1.5); pr = (R.ppf(mp.mpf(q)) + 1) / 2
    print(q, repr(p), mp.nstr(pr, 20), "p abs err %.2e = %.2f ulp(0.5)" % (float(mp.mpf(p) - pr), float((mp.mpf(p) - pr) / mp.mpf(2)**-53)))
    pb = sc.betaincinv(1.5, 1.5, q); print("   betaincinv:", repr(pb), "abs err %.2e" % float(mp.mpf(pb) - pr), " x via betaincinv:", 2*pb-1)
print(scu._beta_ppf(0.5,1.5,1.5), cd.rdist._ppf(np.array([0.5,0.4999999999999999]),3))
