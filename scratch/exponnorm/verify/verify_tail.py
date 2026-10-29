# Left-tail reference, independent of the closed form:
# cdf(x) = int_{-inf}^x phi(s) * (1 - exp(-(x - s)/K)) ds   (Z + T <= x  <=>  T <= x - Z)
# Integrand lives in a layer of width ~1/|x| below s = x, so split finely there.
import mpmath as mp
import scipy.stats as st

def cdf_layer(x, K):
    x = mp.mpf(x); K = mp.mpf(K)
    f = lambda s: mp.npdf(s) * (-mp.expm1(-(x - s) / K))
    offs = [0, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 4e-2, 7e-2, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1, 1.5, 2.5, 4, 6, 10]
    pts = sorted(x - mp.mpf(o) for o in offs)
    return mp.quad(f, pts)

cases = [(-37.0, 1e4), (-20.0, 1e8), (-30.0, 10), (-37.0, 1.0), (-37.0, 1.5), (-37.0, 0.999), (-37.0, 0.1), (-10.0, 1.5),
         (-10.0, 1e17), (-1.0, 1e17)]
for x, K in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(cdf_layer(x, K))
    ref = vals[1]
    agree = abs(vals[0] - vals[1]) / abs(ref)
    s = st.exponnorm.cdf(x, K)
    ulp = abs(mp.mpf(s) - ref) / (abs(ref) * mp.mpf(2.220446049250313e-16))
    print(f"cdf({x!r}, {K!r}) scipy={s!r} ref={mp.nstr(ref, 17)} ulp={mp.nstr(ulp, 3)} agree100v200={mp.nstr(agree, 3)}")
