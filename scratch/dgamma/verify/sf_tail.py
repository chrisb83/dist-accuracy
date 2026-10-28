# sf(2000, 1000): 1 - P cancels ~136 digits, so check at 200 and 300 digits,
# plus the upper incomplete gamma directly at 100 and 200.
import mpmath as mp
from verify import sf
from scipy import stats
for d in (200, 300):
    mp.mp.dps = d; print(d, "1-P:", mp.nstr(sf(2000, 1000), 40))
for d in (100, 200):
    mp.mp.dps = d; print(d, "upper:", mp.nstr(mp.gammainc(1000, 2000, mp.inf, regularized=True)/2, 40))
r = sf(2000, 1000); s = stats.dgamma.sf(2000, 1000)
print(repr(s), float(r), float(abs(s-r)/r/mp.mpf(2)**-52))
