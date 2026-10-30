import time
import mpmath as mp
import numpy as np
from scipy import stats
mp.mp.dps = 40
pi = mp.pi
def f_doc(x):
    x = mp.mpf(x)
    return mp.quad(lambda t: mp.exp(-t*mp.log(t) - x*t)*mp.sin(pi*t), list(range(0, 121)) + [mp.inf]) / pi
def f_boost(x, mu=0, c=1):
    x = mp.mpf(x)
    g = lambda t: mp.exp(-c*t) * mp.cos((x-mu)*t + 2*c/pi*t*mp.log(t))
    return mp.quad(g, mp.linspace(0, 60/c, 121) + [mp.inf]) / pi
t0 = time.time()
for x in [-3, -1, 0, 0.5, 1, 3, 10, 30]:
    d, b, s = f_doc(x), f_boost(x), stats.landau.pdf(x)
    rel = (pi/2)*f_doc(pi*x/2 + mp.log(pi/2))
    print(f"x={x:5}: f_doc={mp.nstr(d,10):>14} f_boost={mp.nstr(b,15):>18} scipy={s:.15e} "
          f"(pi/2)f_doc(..)={mp.nstr(rel,15):>18} rel.err boost-vs-scipy={float(abs(b-s)/b):.1e} relation={float(abs(rel-b)/b):.1e}")
# tails
for x in [1e3, 1e5]:
    print("x", x, "scipy sf*x =", stats.landau.sf(x)*x, " 2/pi =", 2/np.pi,
          " scipy pdf*x^2 =", stats.landau.pdf(x)*x**2)
# doc-form tail: f_doc(lam)*lam^2 -> 1 (so sf -> 1/lam)
for lam in [50, 200]:
    print("lam", lam, "f_doc*lam^2 =", mp.nstr(f_doc(lam)*lam**2, 8))
# normalization of doc form
# loc/scale remark: Landau(mu, c) in Wikipedia/Boost form vs scipy loc=mu+2c/pi log c, scale=c
mu, c = 1.0, 3.0
loc = mu + 2*c/np.pi*np.log(c)
for x in [-2, 2, 8]:
    print("x", x, "wiki(mu,c) =", mp.nstr(f_boost(x, mu, c), 15), " scipy =", stats.landau.pdf(x, loc=loc, scale=c))
print("time", time.time()-t0)
