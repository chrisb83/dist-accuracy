import sys, time; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/ncx2/verify')
import mpmath as mp
from vref import *
from scipy.stats import ncx2
def sf_series2(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    s = 0; j = 0; prev = 0
    while True:
        t = mp.exp(-lam/2 + j*mp.log(lam/2) - mp.loggamma(j+1)) * mp.gammainc(k/2+j, x/2, mp.inf, regularized=True)
        s += t
        if j > lam/2 and t < prev and t < s*mp.mpf(10)**(-mp.mp.dps-5): break
        prev = t; j += 1
    return s
def solve(F, target, a, b):
    g = lambda u: mp.log(F(mp.exp(u))) - mp.log(target)
    return mp.exp(mp.findroot(g, (mp.log(a), mp.log(b)), solver='anderson', tol=mp.mpf(10)**(-mp.mp.dps+10)))
which = sys.argv[1]
cases = {
 'isf': ('F6 isf(1e-300,10,300)', lambda: ncx2.isf(1e-300,10,300), lambda: solve(lambda x: sf_series2(x,10,300), mp.mpf(1e-300), mp.mpf(2900), mp.mpf(3000))),
 'ppf8': ('F8 ppf(0.9999999999999627,10,300)', lambda: ncx2.ppf(0.9999999999999627,10,300), lambda: solve(lambda x: sf_series2(x,10,300), 1-mp.mpf(0.9999999999999627), mp.mpf(620), mp.mpf(630))),
}
name, s, fn = cases[which]
s = float(s()); r = []
for d in (100, 200):
    mp.mp.dps = d; t0 = time.time(); r.append(fn()); print(d, time.time()-t0, flush=True)
mp.mp.dps = 100
ag = abs(r[0]-r[1]) <= abs(r[1])*mp.mpf(10)**-30
print(f"{name}: scipy={s!r} ref={mp.nstr(r[1],17)} agree={ag} ulp={abs(s-float(r[1]))/float(r[1])/2.2e-16:.3g}", flush=True)
