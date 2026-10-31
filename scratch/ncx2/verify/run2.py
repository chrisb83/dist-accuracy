import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/ncx2/verify')
import mpmath as mp
from vref import *
from scipy.stats import ncx2
def q(f, pts, d):
    mp.mp.dps = d
    return mp.quad(f, pts, maxdegree=12)
for d in (100, 200):
    mp.mp.dps = d
    f = lambda t: pdf_bessel(t, 21, 1000)
    pts = [mp.mpf(4000) + i*25 for i in range(0, 41)] + [mp.inf]
    print('sf', d, mp.nstr(mp.quad(f, pts), 40))
    pts = [mp.mpf(i)*5 for i in range(0, 11)]
    print('cdf50', d, mp.nstr(mp.quad(f, pts), 40))
    g = lambda t: pdf_bessel(t, 2, mp.mpf('1e-10'))
    x = mp.mpf('1e-160')
    print('cdf1e-160', d, mp.nstr(mp.quad(g, mp.linspace(0, x, 5)), 40))
    # series route too
    print('cdfser1e-160', d, mp.nstr(cdf_series(x, 2, mp.mpf('1e-10')), 40))
try: print(ncx2.sf(1e-10, 21, 1000))
except Exception as e: print('F2 scipy raises', type(e).__name__, e)
