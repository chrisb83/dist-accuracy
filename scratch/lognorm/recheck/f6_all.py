import re, numpy as np, mpmath as mp
from scipy import special as sc
mp.mp.dps = 80; eps = 2.0**-52
txt = open('../run1.txt').read().split('**cdf** (27 points)')[1].split('**logcdf**')[0]
rows = re.findall(r'^\| \(([^)]*)\) \| ([^ |]+) \|', txt, re.M)
S_spec = S_arg = 0; nspec = 0
print("s x z total ndtr arg")
for s, x in rows:
    s = float(s); x = float(x)
    if s == 0.953682: s = 0.9536822696057533
    zd = np.log(x)/s; ze = mp.log(mp.mpf(x))/mp.mpf(s)
    r = mp.ncdf(ze); rd = mp.ncdf(mp.mpf(zd))
    tot = float(abs(mp.mpf(sc.ndtr(zd))-r)/r/eps); spec = float(abs(mp.mpf(sc.ndtr(zd))-rd)/r/eps); arg = float(abs(rd-r)/r/eps)
    nspec += spec > arg
    print(f"{s:g} {x!r} z={zd:.3f} total={tot:.0f} ndtr={spec:.0f} arg={arg:.0f}")
print("points where ndtr part > arg part:", nspec, "of", len(rows))
