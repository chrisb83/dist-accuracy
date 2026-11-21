import sys; sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy import stats
for chi in [1.4143, 1.5, 2, 3, 10, 100]:
    a = chi*chi/2; xb = np.sqrt(1 - 1/a)
    x = np.linspace(xb*(1 - 1e-6), min(xb*(1 + 1e-6), 1 - 1e-16), 20001)
    c = stats.argus.cdf(x, chi); s = stats.argus.sf(x, chi)
    print(f"chi={chi}: cdf monotone {np.all(np.diff(c) >= 0)}, max |cdf+sf-1| {np.max(np.abs(c + s - 1)):.1e}")
s2 = np.sqrt(2); chis = s2*(1 + np.linspace(-1e-12, 1e-12, 2001))
for m in ['logpdf', 'sf', 'cdf']:
    v = getattr(stats.argus, m)(0.5, chis)
    print(m, "rel step across chi = sqrt(2):", f"{np.max(np.abs(np.diff(v)/v[1:])):.1e}")
