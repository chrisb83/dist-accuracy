import mpmath as mp
import numpy as np
from scipy import stats

def pdf(x):
    x = mp.mpf(x)
    return mp.exp(-(x-1)**2/(2*x))/mp.sqrt(2*mp.pi*x**3)

def logpdf(x):
    x = mp.mpf(x)
    return -mp.log(2*mp.pi)/2 - 3*mp.log(x)/2 - (x-1)**2/(2*x)

def sf_quad(x):
    # integrate pdf(x+u)/pdf(x) (O(1) integrand; mp.quad uses an absolute tolerance)
    x = mp.mpf(x); p0 = pdf(x)
    pts = [0] + [mp.mpf(2)**k for k in range(-4, 12)] + [mp.inf]
    return p0*mp.quad(lambda u: pdf(x+u)/p0, pts)

def sf_erfc(x):
    # complementary route: sf = (erfc(z1/r2) - e^2 erfc(z2/r2))/2
    x = mp.mpf(x); s = mp.sqrt(x)
    z1 = (x-1)/s; z2 = (x+1)/s
    return (mp.erfc(z1/mp.sqrt(2)) - mp.exp(2)*mp.erfc(z2/mp.sqrt(2)))/2

def isf(q):
    q = mp.mpf(q)
    lq = mp.log(q)
    x0 = -2*lq
    return mp.findroot(lambda x: mp.log(sf_erfc(x)) - lq, x0)

calls = [
 ("F1", "isf", 1e-300), ("F1", "isf", 1e-100), ("F1", "isf", 1e-70),
 ("F1", "isf", 1e-67), ("F1", "isf", 1e-65),
 ("F2", "logpdf", 1.4e154), ("F2", "logpdf", 1e200),
 ("F3", "logcdf", 1000.0), ("F3", "sf", 1000.0), ("F3", "sf", 300.0), ("F3", "sf", 100.0),
]

def ref(m, a):
    if m == "isf":
        x = isf(a)
        chk = sf_quad(x) / mp.mpf(a) - 1  # quantile check via quadrature
        return x, chk
    if m == "logpdf":
        return logpdf(a), None
    if m == "sf":
        v = sf_quad(a); return v, sf_erfc(a)/v - 1
    if m == "logcdf":
        v = mp.log1p(-sf_quad(a)); return v, None

def main():
  out = {}
  for dps in (100, 200):
      mp.mp.dps = dps
      out[dps] = [ref(m, a) for _, m, a in calls]

  mp.mp.dps = 50
  for (fid, m, a), (r1, c1), (r2, _) in zip(calls, out[100], out[200]):
      agree = abs(r1 - r2) / abs(r2)
      s = getattr(stats.wald, m)(a)
      err = abs(mp.mpf(float(s)) - r2)/abs(r2)/mp.mpf(2.220446049250313e-16) if np.isfinite(s) else mp.inf
      print(f"{fid} {m}({a!r}) scipy={s!r} ref={mp.nstr(r2, 17)} ulp={mp.nstr(err, 3)} "
            f"agree100v200={mp.nstr(agree, 3)} check={mp.nstr(c1, 3) if c1 is not None else '-'}")

if __name__ == "__main__":
    main()
