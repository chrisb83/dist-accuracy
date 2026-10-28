import sys
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/fisk/verify')
from mpmath import mp
import ref
import numpy as np
from scipy.stats import fisk

C = 3.085754862225318
cases = [
 ('sf', 1e10, C), ('logsf', 1e10, C), ('sf', 1.5, 100.0), ('logsf', 1.5, 100.0),
 ('sf', 1e20, 1.0), ('sf', 1e30, 0.5), ('logsf', 1e30, 0.5), ('sf', 1e10, 1.2),
 ('sf', 9.0, C),
 ('pdf', 1e-300, 2.0), ('pdf', 1e-140, C), ('pdf', 1e-300, 0.5),
 ('logcdf', 1e-300, C), ('logpdf', 1e-300, C), ('logcdf', 1e-300, 1e6),
 ('logpdf', 1e-300, 1.0), ('logpdf', 1e-10, 1.0), ('logpdf', 1e-130, 1.0000000001),
]
def getref(dps, m, x, c):
    mp.dps = dps
    return getattr(ref, m)(x, c)

print(f"{'call':42s} {'scipy':26s} {'ref':26s} {'ulp':>12s}")
for m, x, c in cases:
    r100 = getref(100, m, x, c); r200 = getref(200, m, x, c)
    mp.dps = 220
    agree = mp.nstr(abs(r100-r200)/abs(r200) if r200 != 0 else abs(r100-r200), 3)
    mp.dps = 200
    sv = float(getattr(fisk, m)(x, c))
    rf = float(r200)
    if rf == 0 or not np.isfinite(sv):
        ulp = float('inf') if sv != rf else 0.0
    else:
        ulp = float(abs(mp.mpf(sv)-r200)/abs(r200)/mp.mpf('2.2e-16'))
    print(f"{m+'('+repr(x)+', '+repr(c)+')':42s} {sv!r:26s} {rf!r:26s} {ulp:12.3g}  reldiff100v200={agree}")
