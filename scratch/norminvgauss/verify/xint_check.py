# independent check: integrate SciPy's (accurate here) pdf in u = asinh(x) with tight tolerances
import numpy as np, warnings
from scipy import integrate, stats
warnings.simplefilter('ignore')
pdf = stats.norminvgauss._pdf
a, b = 10, 9.99999
for x in [67.445903186346489, 67.44589577432565]:
    g = lambda u: pdf(np.sinh(u), a, b) * np.cosh(u)
    U = np.arcsinh(x)
    pts = np.linspace(-20, 40, 241)
    lo = sum(integrate.quad(g, p, q, epsabs=0, epsrel=1e-13, limit=200)[0] for p, q in zip(pts[:-1], pts[1:]) if q <= U)
    lo += integrate.quad(g, max(p for p in pts if p <= U), U, epsabs=0, epsrel=1e-13)[0]
    hi = sum(integrate.quad(g, p, q, epsabs=0, epsrel=1e-13, limit=200)[0] for p, q in zip(pts[:-1], pts[1:]) if p >= U)
    hi += integrate.quad(g, U, min(p for p in pts if p >= U), epsabs=0, epsrel=1e-13)[0]
    print(x, 'cdf', lo, 'sf', hi, 'sum', lo + hi)
