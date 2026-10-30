"""One demonstration point per inherited method (generic default vs mp.quad of the reference pdf)."""
import sys, warnings
import mpmath as mp, numpy as np
from scipy import stats
sys.path.insert(0, r"C:\Users\chbau\Documents\research\dist-accuracy\refs")
sys.path.insert(0, r"C:\Users\chbau\Documents\research\dist-accuracy\.claude\skills\dist-accuracy-review\scripts")
import gausshyper_ref as R
from accuracy import invert
warnings.simplefilter("ignore")
mp.mp.dps = 40
s = (2.0, 3.0, 1.0, 0.5)
P = [mp.mpf(v) for v in s]
f = lambda t: R.pdf(t, *P)
cdf = lambda x: mp.quad(f, [0, x])
sf = lambda x: mp.quad(f, [x, 1])
u = lambda v, r: float(abs(mp.mpf(v) / r - 1) / 2.22e-16) if r != 0 else float("nan")
x = 1 - 1e-6
r = sf(mp.mpf(x)); v = stats.gausshyper.sf(x, *s); print("sf", x, v, mp.nstr(r, 17), u(v, r))
r = mp.log(sf(mp.mpf(x))); v = stats.gausshyper.logsf(x, *s); print("logsf", x, v, mp.nstr(r, 17), u(v, r))
x = 1e-5
r = cdf(mp.mpf(x)); v = stats.gausshyper.cdf(x, *s); print("cdf", x, v, mp.nstr(r, 17), u(v, r))
x = 1e-200
r = mp.log(cdf(mp.mpf(x))); v = stats.gausshyper.logcdf(x, *s); print("logcdf", x, v, mp.nstr(r, 17), u(v, r))
q = 1e-20
v = stats.gausshyper.ppf(q, *s)
r = invert(lambda t: cdf(t) if 0 < t < 1 else (mp.mpf(0) if t <= 0 else mp.mpf(1)),
           mp.mpf(q), lo=mp.mpf("1e-12"), hi=mp.mpf("0.5"), positive=True, digits=25)
print("ppf", q, v, mp.nstr(r, 17), u(v, r))
v = stats.gausshyper.isf(q, *s)
y = invert(lambda t: sf(1 - t) if 0 < t < 1 else (mp.mpf(0) if t <= 0 else mp.mpf(1)),
           mp.mpf(q), lo=mp.mpf("1e-12"), hi=mp.mpf("0.5"), positive=True, digits=25)
print("isf", q, v, "ref 1 - x =", mp.nstr(y, 17), "scipy 1 - x =", 1 - v)
