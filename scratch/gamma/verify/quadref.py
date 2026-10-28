"""Independent reference for the regularized lower incomplete gamma P(a,x)
for x < a and large a, by high-precision quadrature of the gamma pdf.

P(a,x) = int_0^x t^(a-1) e^-t dt / Gamma(a).
Substitution t = x - sqrt(a)*v  (integrand peaks at the right endpoint t=x
because x < a-1), so the whole mass sits in v in [0, ~30].
This route uses NO series for the incomplete gamma: only mp.loggamma,
mp.log, mp.exp and mp.quad (tanh-sinh / Gauss-Legendre).
"""
from mpmath import mp


def P_quad(a, x, V=60, dps=100, maxdegree=12):
    mp.dps = dps + 20
    a = mp.mpf(a); x = mp.mpf(x)
    sq = mp.sqrt(a)
    lg = mp.loggamma(a)

    def f(v):
        t = x - sq * v
        if t <= 0:
            return mp.mpf(0)
        return mp.e ** ((a - 1) * mp.log(t) - t - lg) * sq

    # split points so each panel is smooth
    pts = [mp.mpf(0), mp.mpf(1), mp.mpf(2), mp.mpf(4), mp.mpf(8),
           mp.mpf(16), mp.mpf(32), mp.mpf(V)]
    pts = [p for p in pts if p <= V]
    if pts[-1] != V:
        pts.append(mp.mpf(V))
    val = mp.quad(f, pts, maxdegree=maxdegree)
    mp.dps = dps
    return +val


if __name__ == "__main__":
    import sys, time
    cases = [(1e6, 995000.0), (1e6, 992000.0), (1e6, 990000.0),
             (1e8, 99954000.0), (1e8, 99900000.0)]
    for a, x in cases:
        for dps in (100, 200):
            t0 = time.time()
            v = P_quad(a, x, dps=dps)
            print("a=%g x=%.10g dps=%d  %s   %.1fs" % (a, x, dps, mp.nstr(v, 40), time.time()-t0))
