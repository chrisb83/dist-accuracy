import sys, warnings; sys.path.insert(0, '.')
import mpmath as mp, numpy as np
from scipy import stats
from vref import pdf, cdf, sf, both
warnings.simplefilter('ignore')
r, ok = both(cdf, 1e308, 1, 0.01, 1); print('F8 cdf ref', mp.nstr(r, 17), ok, 'scipy', stats.ncf.cdf(1e308,1,0.01,1))
r, ok = both(sf, 8e307, 5, 0.5, 2); print('F8 sf(8e307,5,.5,2) ref', mp.nstr(r, 17), ok, 'scipy', stats.ncf.sf(8e307,5,0.5,2))
# different route: quad of pdf on finite interval
mp.mp.dps = 40
print('F3 cdf quad', mp.nstr(mp.quad(lambda u: pdf(u,1,1,1e-20), [0, 1, 10]), 17))
print('F2 cdf quad', mp.nstr(mp.quad(lambda u: pdf(u,1,1,1), [0, mp.mpf(1e-210)]), 17))
# sf F8 via substitution u = 1/x on [0, 1e-308]: sf = int_0^{1/x} pdf(1/u)/u^2 du
print('F8 sf quad', mp.nstr(mp.quad(lambda u: pdf(1/u,1,0.01,1)/u**2, [0, 1/mp.mpf(1e308)]), 17))
