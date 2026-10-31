import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/ncx2/verify')
import mpmath as mp, numpy as np
from vref import *
from scipy.stats import ncx2
def both(fn):
    out = []
    for d in (100, 200):
        mp.mp.dps = d; out.append(fn())
    mp.mp.dps = 100
    agree = abs(out[0]-out[1]) <= abs(out[1])*mp.mpf(10)**-30 or out[1] == 0
    return out[1], agree
def ulp(s, r):
    r = float(r) if not isinstance(r, float) else r
    try: return abs(s - r)/abs(r)/2.2e-16
    except Exception: return None
cases = [
 ('F1 pdf(1e-7,21,60)', lambda: ncx2.pdf(1e-7,21,60), lambda: pdf_pois(mp.mpf('1e-7'),21,60), lambda: pdf_bessel(mp.mpf('1e-7'),21,60)),
 ('F1 cdf(50,21,1000)', lambda: ncx2.cdf(50,21,1000), lambda: cdf_quad(50,21,1000,[25,40]), None),
 ('F1 sf(4000,21,1000)', lambda: ncx2.sf(4000,21,1000), lambda: sf_quad(4000,21,1000,[4050,4200,4600]), None),
 ('F2 sf(1e-10,21,1000)', None, lambda: 1 - cdf_quad(mp.mpf('1e-10'),21,1000), None),
 ('F3 logpdf(1000,1000,1)', lambda: ncx2.logpdf(1000,1000,1), lambda: mp.log(pdf_pois(1000,1000,1)), lambda: mp.log(pdf_bessel(1000,1000,1))),
 ('F4 logpdf(1e20,21,1)', lambda: ncx2.logpdf(1e20,21,1), lambda: mp.log(pdf_bessel(mp.mpf('1e20'),21,1)), None),
 ('F5 logpdf(1e10,2.5,1e-300)', lambda: ncx2.logpdf(1e10,2.5,1e-300), lambda: mp.log(pdf_pois(mp.mpf('1e10'),2.5,mp.mpf('1e-300'))), lambda: mp.log(pdf_bessel(mp.mpf('1e10'),2.5,mp.mpf('1e-300')))),
 ('F7 cdf(1e-160,2,1e-10)', lambda: ncx2.cdf(1e-160,2,1e-10), lambda: cdf_quad(mp.mpf('1e-160'),2,mp.mpf('1e-10')), None),
 ('F9 pdf(12100,3,1e4)', lambda: ncx2.pdf(12100,3,10000), lambda: pdf_pois(12100,3,10000), lambda: pdf_bessel(12100,3,10000)),
]
for name, sp, ref, alt in cases:
    try: s = float(sp()) if sp else None
    except Exception as e: s = repr(e)
    r, ag = both(ref)
    line = f"{name}: scipy={s!r} ref={mp.nstr(r,17)} agree100/200={ag}"
    if alt: 
        a, _ = both(alt); line += f" alt_rel={mp.nstr(abs(a-r)/abs(r),3)}"
    if isinstance(s, float): line += f" ulp={ulp(s, r)}"
    print(line, flush=True)
