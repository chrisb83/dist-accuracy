# Independent verifier references for genlogistic (standard form).
# pdf from docstring: f = c e^{-x} / (1+e^{-x})^{c+1}; here evaluated in the
# equivalent form c e^{cx} / (1+e^x)^{c+1}. cdf/sf by mp.quad of the pdf,
# ppf/isf by closed form, then checked via cdf(x_ref) == q.
import warnings, mpmath as mp, numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

def logpdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    return mp.log(c) + c*x - (c+1)*mp.log1p(mp.exp(x))
def pdf(x, c): return mp.exp(logpdf(x, c))

def bps(x, c, side):
    c = mp.mpf(c); x = mp.mpf(x)
    pts = {x, mp.log(c)} | {mp.log(c) + s*10**k for k in range(-3, 3) for s in (-1, 1)}
    pts |= {x + (-1 if side == 'L' else 1)*mp.mpf(10)**k for k in range(-4, 13)}
    pts |= {(-1 if side == 'L' else 1)*mp.mpf(10)**k / c for k in range(-2, 4)}
    pts = sorted(p for p in pts if (p < x if side == 'L' else p > x))
    return ([-mp.inf] + pts + [x]) if side == 'L' else ([x] + pts + [mp.inf])

def cdf_quad(x, c): return mp.quad(lambda y: pdf(y, c), bps(x, c, 'L'))
def sf_quad(x, c):  return mp.quad(lambda y: pdf(y, c), bps(x, c, 'R'))
def cdf_closed(x, c):  # logistic(x)**c, logistic in left-stable form
    x, c = mp.mpf(x), mp.mpf(c)
    return mp.exp(c*(x - mp.log1p(mp.exp(x))))
def ppf(q, c):
    q, c = mp.mpf(q), mp.mpf(c)
    return -mp.log(mp.expm1(-mp.log(q)/c))
def isf(q, c): return ppf(1 - mp.mpf(q), c)

CASES = [
 ("F1","cdf",-710,0.001),("F1","cdf",-1000,0.41192440799679475),("F1","cdf",-1e10,1e-10),
 ("F1","logcdf",-710,1),("F1","logcdf",-1000,0.001),("F1","logcdf",-1000,0.41192440799679475),
 ("F1","logcdf",-1e300,0.41192440799679475),
 ("F2","sf",-710,1e-20),("F2","sf",-1000,1e-10),("F2","sf",-1e7,1e-10),("F2","sf",-710,0.001),("F2","sf",-1000,0.001),
 ("F3","logpdf",-1e300,1e-20),("F3","logpdf",-1e10,1e-10),("F3","pdf",-1e22,1e-20),("F3","pdf",-1e20,1e-20),
 ("F3","pdf",-1e12,1e-10),("F3","pdf",-1e10,1e-10),
 ("F4","ppf",1e-300,0.41192440799679475),("F4","ppf",1e-5,0.01),("F4","ppf",0.5,1e-20),("F4","isf",0.5,1e-10),
 ("F5","isf",1e-17,1),("F5","isf",1e-300,50),("F5","isf",1e-16,1),("F5","isf",1e-12,1),("F5","isf",1e-8,1),("F5","isf",1e-8,1e-10),
 ("F6","cdf",34.538776394910684,1e15),("F6","cdf",30,1e15),("F6","cdf",8,1e6),("F6","cdf",13.815510557964274,1e6),
 ("F7","ppf",0.49999999999999994,1),("F7","isf",0.49999999999999994,1),
]

def ref(m, a, c):
    if m == "pdf": return pdf(a, c), None
    if m == "logpdf": return logpdf(a, c), None
    if m == "cdf":
        v = cdf_quad(a, c) if abs(a) < 1e8 else None
        return cdf_closed(a, c), v
    if m == "logcdf":
        v = mp.log(cdf_quad(a, c)) if abs(a) < 1e8 else None
        return mp.log(cdf_closed(a, c)), v
    if m == "sf":
        return 1 - cdf_closed(a, c), sf_quad(a, c)
    if m == "ppf":
        x = ppf(a, c); return x, cdf_closed(x, c) / mp.mpf(a) - 1  # inversion residual
    if m == "isf":
        x = isf(a, c); return x, (1 - cdf_closed(x, c)) / mp.mpf(a) - 1

if __name__ == "__main__":
 for fid, m, a, c in CASES:
      out = []
      for d in (100, 200):
          mp.mp.dps = d
          out.append(ref(m, a, c))
      mp.mp.dps = 100
      r1, r2 = out[0][0], out[1][0]
      stable = mp.almosteq(r1, r2, rel_eps=mp.mpf(10)**-30) if r2 != 0 else r1 == 0
      alt = out[0][1]
      if m in ("ppf", "isf"):
          altinfo = "resid=%s" % mp.nstr(alt, 3)
      elif alt is None:
          altinfo = "quad n/a"
      else:
          altinfo = "quad relerr=%s" % mp.nstr(abs(alt/r1 - 1) if r1 != 0 else alt, 3)
      s = getattr(stats.genlogistic, m)(a, c)
      rf = float(r1)
      err = abs(s - rf) / abs(rf) / 2.220446049250313e-16 if rf != 0 else float('nan')
      print(f"{fid} {m}({a!r}, {c!r}): scipy={s!r} ref={mp.nstr(r1, 17)} ulp={err:.3g} stable30={stable} {altinfo}")
