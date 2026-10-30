import numpy as np, mpmath as mp
from scipy import special as sc
mp.mp.dps = 100
eps = 2.0**-52
def ulp(a, r):
    r = mp.mpf(r)
    return float(abs(mp.mpf(a) - r) / (abs(r) * eps)) if r != 0 else float('nan')
# (method, x, s)
pts = [("cdf",1e-160,10.),("sf",1e160,10.),("logcdf",1e160,10.),("logsf",1e-160,10.),
       ("cdf",1e-10,1.),("cdf",9.357622968840175e-14,1.),("cdf",0.00033546262790251185,1.),
       ("cdf",1e-140,10.),("cdf",1e-100,10.),("cdf",1e-40,10.)]
# s=1e-3 tail: x = exp(-30*1e-3), exp(-37.5e-3)
for zz in (-8., -20., -30., -37.5):
    pts.append(("cdf", float(np.exp(zz*1e-3)), 1e-3))
print("method x s | z_dbl | total ulp | ndtr-at-zdbl ulp | arg-rounding ulp (F(zdbl) vs F(z))  | rel err of z")
for m, x, s in pts:
    zd = np.log(x)/s
    ze = mp.log(mp.mpf(x))/mp.mpf(s)
    if m == "cdf":   f = lambda z: mp.ncdf(z); sc_ = sc.ndtr(zd)
    if m == "sf":    f = lambda z: mp.ncdf(-z); sc_ = sc.ndtr(-zd)
    if m == "logcdf":f = lambda z: mp.log1p(-mp.ncdf(-z)); sc_ = sc.log_ndtr(zd)
    if m == "logsf": f = lambda z: mp.log1p(-mp.ncdf(z)); sc_ = sc.log_ndtr(-zd)
    tot = ulp(sc_, f(ze)); spec = ulp(sc_, f(mp.mpf(zd)))
    arg = float(abs(f(mp.mpf(zd)) - f(ze))/abs(f(ze))/eps)
    signed_arg = float((f(mp.mpf(zd)) - f(ze))/abs(f(ze))/eps)
    signed_spec = float((mp.mpf(sc_) - f(mp.mpf(zd)))/abs(f(ze))/eps)
    print(f"{m}({x!r}, {s}) z={zd!r} total={tot:.4g} special={spec:.4g} ({signed_spec:+.4g}) arg={arg:.4g} ({signed_arg:+.4g}) dz/z={float((mp.mpf(zd)-ze)/ze/eps):+.3g} eps")
