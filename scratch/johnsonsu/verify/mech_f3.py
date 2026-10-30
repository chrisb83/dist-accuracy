import numpy as np, mpmath as mp
from scipy import special as sc
from verify import zq_lower, ppf
mp.mp.dps = 100
q, a, b = 4.906713932054622e-198, -30.0, 0.5
z = zq_lower(q); zd = float(z); zs = sc.ndtri(q)
ref = ppf(q, a, b)
print("ndtri ulp err", float((mp.mpf(zs)-z)/(abs(z)*2**-52)))
for name, zz in [("correctly rounded", zd), ("scipy ndtri", zs)]:
    x = np.sinh((zz - a)/b)
    print(name, x, float(abs(x-ref)/abs(ref)))
