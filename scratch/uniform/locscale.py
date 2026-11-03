import numpy as np, mpmath as mp, warnings
from scipy import stats
mp.mp.dps = 800
U = stats.uniform
def ulp(c, r):
    r = mp.mpf(r)
    if r == 0: return 0.0 if c == 0 else float('inf')
    return float(abs(mp.mpf(c) - r) / abs(r) / mp.mpf(2)**-52)
def show(name, c, r, cond):
    print(f"{name:55s} scipy={c!r:24} ref={mp.nstr(r,17):24} ulp={ulp(c,r):.3g} cond={cond}")
# inherited methods, standard form
for x in [1e-17, 1e-300, 1 - 2**-52, 0.5]:
    xm = mp.mpf(x)
    show(f"logsf({x!r})", U.logsf(x), mp.log(1 - xm), mp.nstr(abs(xm/((1-xm)*mp.log(1-xm))),3))
    show(f"sf({x!r})", U.sf(x), 1 - xm, mp.nstr(xm/(1-xm),3))
for q in [1e-300, 1e-17, 1 - 2**-52]:
    show(f"isf({q!r})", U.isf(q), 1 - mp.mpf(q), mp.nstr(abs(mp.mpf(q)/(1-mp.mpf(q))),3))
# loc/scale
cases = [("cdf", 1e16, 1e16, 1.0), ("cdf", 1e16+2, 1e16, 4.0), ("sf", 1e16+2, 1e16, 4.0),
         ("cdf", 3e-300, 1e-300, 1e-299), ("cdf", 1.5e308, -1e308, 1.7e308), ("sf", 1.5e308, -1e308, 1.7e308),
         ("cdf", 7e-310, 0.0, 1e-309), ("pdf", 0.5e308, 0.0, 1.7e308), ("pdf", 5e-310, 0.0, 1e-309),
         ("logpdf", 5e-310, 0.0, 1e-309),
         ("ppf", 0.3, -1.0, 3.0), ("ppf", 1/3, -1.0, 3.0), ("ppf", 1e-300, 1.0, 1.0), ("isf", 1e-17, 2.0, 3.0),
         ("ppf", 0.9, -1e308, 1.7e308), ("ppf", 0.5, 1e16, 1.0)]
for m, a, loc, scale in cases:
    A, L, S = mp.mpf(a), mp.mpf(loc), mp.mpf(scale)
    if m in ("cdf", "sf", "pdf", "logpdf"):
        z = (A - L) / S
        F = min(max(z, 0), 1)
        r = {"cdf": F, "sf": 1 - F, "pdf": (1/S if 0 <= z <= 1 else 0), "logpdf": -mp.log(S)}[m]
    else:
        r = L + S * (A if m == "ppf" else 1 - A)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = getattr(U, m)(a, loc=loc, scale=scale)
    show(f"{m}({a!r}, loc={loc!r}, scale={scale!r})", c, r, "-" + (f" WARN {w[0].message}" if w else ""))
