# Layer integral with explicit Gaussian (no mp.npdf), variable d = x - s >= 0:
# cdf(x) = phi(x) * int_0^inf exp(x*d - d^2/2) * (-expm1(-d/K)) dd   (x < 0)
import mpmath as mp
import scipy.stats as st
def layer(x, K):
    x = mp.mpf(x); K = mp.mpf(K)
    g = lambda d: mp.exp(x*d - d*d/2) * (-mp.expm1(-d/K))
    a = 1/abs(x)
    pts = [mp.mpf(0)] + [a*m for m in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)] + [mp.mpf(60)]
    pts = sorted(set(pts))
    v, e = mp.quad(g, pts, error=True)
    return mp.exp(-x*x/2)/mp.sqrt(2*mp.pi) * v, e/v
mp.mp.dps = 30
print("npdf check", mp.npdf(-37), mp.exp(mp.mpf(-37)**2/-2)/mp.sqrt(2*mp.pi))
cases = [(-37.0, 1e4), (-20.0, 1e8), (-30.0, 10), (-37.0, 1.0), (-37.0, 1.5), (-37.0, 0.999), (-37.0, 0.1), (-10.0, 1.5)]
for x, K in cases:
    out = []
    for dps in (100, 200):
        mp.mp.dps = dps
        out.append(layer(x, K))
    ref = out[1][0]
    agree = abs(out[0][0] - ref)/ref
    s = st.exponnorm.cdf(x, K)
    ulp = abs(mp.mpf(s) - ref)/(ref*mp.mpf(2.220446049250313e-16))
    print(f"cdf({x!r}, {K!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={mp.nstr(ulp,3)} agree={mp.nstr(agree,3)} quaderr={mp.nstr(out[1][1],3)}")
