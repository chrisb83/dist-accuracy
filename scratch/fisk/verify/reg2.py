import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/fisk/verify')
import numpy as np, warnings
from mpmath import mp
import ref
from scipy.stats import fisk
warnings.simplefilter('ignore')
mp.dps = 120
for c in [1.0, 1.2, 3.085754862225318, 1e6]:
    on_lc = on_lp = None
    for e in range(-1, -309, -1):
        if on_lc is None and not np.isfinite(fisk.logcdf(10.0**e, c)): on_lc = e
        if on_lp is None and not np.isfinite(fisk.logpdf(10.0**e, c)): on_lp = e
    print(f"c={c}: logcdf -inf from 1e{on_lc}, logpdf -inf from 1e{on_lp}, predicted 1e{-308.25/c:.1f}")
print()
for c in [1.0, 1+1e-10, 1+1e-8, 1+1e-6, 1.001]:
    for x in [1e-10, 1e-130]:
        s = fisk.logpdf(x, c); r = ref.logpdf(x, c)
        ulp = abs(mp.mpf(s)-r)/abs(r)/mp.mpf('2.2e-16')
        print(f"logpdf({x:g},{c!r}): scipy={s!r} ref={float(r)!r} ulp={float(ulp):.3g}")
