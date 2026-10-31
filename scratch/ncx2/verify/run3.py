import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/ncx2/verify')
import mpmath as mp
from vref import *
def sf_series(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    s = 0; j = 0
    while True:
        t = mp.exp(-lam/2 + j*mp.log(lam/2) - mp.loggamma(j+1)) * mp.gammainc(k/2+j, x/2, mp.inf, regularized=True)
        s += t
        if j > x and t < s*mp.mpf(10)**(-mp.mp.dps-5): break
        j += 1
    return s
for d in (100, 200):
    mp.mp.dps = d
    print('sf series', d, mp.nstr(sf_series(4000, 21, 1000), 40))
    print('cdf50 series', d, mp.nstr(cdf_series(50, 21, 1000), 40))
    f = lambda t: pdf_bessel(t, 21, 1000)
    v, e = mp.quad(f, [mp.mpf(4000) + i*5 for i in range(0, 401)] + [mp.inf], error=True)
    print('sf quad fine', d, mp.nstr(v, 40), mp.nstr(e, 3))
