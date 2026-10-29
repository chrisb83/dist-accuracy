# Independent reference: cdf as convolution E[Phi(x - T)], T ~ Exp(mean K),
# cdf(x) = int_0^inf (1/K) exp(-t/K) Phi(x - t) dt  (mp.quad), not the closed form.
import mpmath as mp
import scipy.stats as st

def Phi(z):
    return mp.erfc(-z / mp.sqrt(2)) / 2

def cdf_conv(x, K):
    x = mp.mpf(x); K = mp.mpf(K); lam = 1 / K
    f = lambda t: lam * mp.exp(-lam * t) * Phi(x - t)
    c = max(x, mp.mpf(0))
    pts = [mp.mpf(0)]
    if c > 0:
        pts += [c - 50, c - 10] if c > 60 else []
        pts += [c]
    pts += [c + 1, c + 3, c + 7, c + 15, c + 25, c + 40, c + 70, c + 100]
    pts = sorted(set(p for p in pts if p >= 0))
    return mp.quad(f, pts)

def cdf_closed_check(x, K):
    # second, independent route: integrate the docstring pdf from -inf to x
    x = mp.mpf(x); K = mp.mpf(K)
    pdf = lambda s: 1/(2*K)*mp.exp(1/(2*K**2) - s/K)*mp.erfc(-(s - 1/K)/mp.sqrt(2))
    return mp.quad(pdf, [x - 100, x - 40, x - 20, x - 5, x - 1, x])

cases = [(-1.0, 1e17), (-10.0, 1e17), (1.0, 1e17), (1e10, 1e17), (1.0, 1e8), (0.0, 1e4),
         (-37.0, 1e4), (-20.0, 1e8), (-30.0, 10), (-37.0, 1.0), (-37.0, 1.5), (-37.0, 0.999), (-37.0, 0.1)]
for x, K in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(cdf_conv(x, K))
    mp.mp.dps = 100
    agree = abs(vals[0] - vals[1]) / abs(vals[1])
    alt = cdf_closed_check(x, K) if K <= 1e4 else None
    ref = vals[1]
    s = st.exponnorm.cdf(x, K)
    ulp = abs(mp.mpf(s) - ref) / (abs(ref) * mp.mpf(2.220446049250313e-16))
    print(f"cdf({x!r}, {K!r}) scipy={s!r} ref={mp.nstr(ref, 17)} ulp={mp.nstr(ulp, 3)} "
          f"agree100v200={mp.nstr(agree, 3)} pdfint_rel={mp.nstr(abs(alt-ref)/abs(ref),3) if alt is not None else '-'}")
