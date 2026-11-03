"""cdf near x = 0 for large c: is the error from rounding y = (x+1)/2?"""
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, scipy.special as sc
import rdist_ref as r
mp.mp.dps = 120
for c, x in [(1e6, 1e-8), (1e6, -1e-8), (1e6, 9.99999500000375e-06), (1e8, 1e-8), (1e8, 3e-9), (1e8, 1e-7)]:
    a = c / 2
    y = (x + 1) / 2
    yerr = mp.mpf(y) - (1 + mp.mpf(x)) / 2
    ref = r.cdf(mp.mpf(x), c)
    at_rounded_y = r._I_low(mp.mpf(y), mp.mpf(a)) if y <= 0.5 else 1 - r._I_low(1 - mp.mpf(y), mp.mpf(a))
    got = sc.betainc(a, a, y)
    u = lambda v: float(abs((mp.mpf(v) - ref) / ref)) / 2.22e-16
    print(f"c={c:g} x={x!r}: y rounding {float(yerr):.2e}; exact I at rounded y {u(at_rounded_y):.3g} ulp; scipy {u(got):.3g} ulp;"
          f" Boost vs exact-at-rounded-y {float(abs((mp.mpf(got)-at_rounded_y)/at_rounded_y))/2.22e-16:.3g} ulp")
