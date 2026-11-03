"""cdf tails at c = 1e6: rounding of y = (x+1)/2 vs Boost error (exact I at the rounded y)."""
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, numpy as np, scipy.special as sc
import rdist_ref as r
mp.mp.dps = 120
for c, x in [(1e6, -0.029999985000011248), (1e6, -0.0199999900000075), (1e6, -0.00999999500000375),
             (1e6, -0.00199999900000075), (1e6, -0.000999999500000375), (1e3, -0.3), (1e3, -0.29999999999999993)]:
    a = c / 2; y = (x + 1) / 2
    ref = r.cdf(mp.mpf(x), c)
    exact_at_y = r._I_low(mp.mpf(y), mp.mpf(a))
    got = sc.betainc(a, a, y)
    cand = 0.5 + 0.5 * np.sign(x) * sc.betainc(0.5, a, x * x) if x > -0.01 else 0.5 * sc.betaincc(0.5, a, x * x)
    u = lambda v, rr: float(abs((mp.mpf(v) - rr) / rr)) / 2.22e-16
    print(f"c={c:g} x={x!r}: y err {float(mp.mpf(y) - (1 + mp.mpf(x)) / 2):.2e}; exact I(rounded y) {u(exact_at_y, ref):.3g} ulp;"
          f" Boost vs exact I(rounded y) {u(got, exact_at_y):.3g} ulp; scipy {u(got, ref):.3g} ulp; X^2-form {u(cand, ref):.3g} ulp")
