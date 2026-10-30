# brute-force check of the left tail: same integrand, uniform fine subdivision
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, landau_ref as R, time
from scipy import stats
for dps in (100, 200):
    mp.mp.dps = dps
    for x in (-4.8225530837815455631, -5.0):
        X = mp.mpf(x)
        t = time.time()
        p, c = R.pdf(X), R.cdf(X)
        t1 = time.time() - t
        with mp.workdps(dps + 10):
            y = R._y(X)
            s0 = mp.exp(-y - 1)
            w = 1 / mp.sqrt(s0)              # gaussian width in theta
            pts = [min(w * k, mp.mpf(3)) for k in range(0, 41)]
            pts = sorted(set(pts))
            bf = mp.quad(lambda t: mp.exp(-mp.exp(R._vt(t, y))), pts) / mp.pi
            bp = mp.quad(lambda t: (lambda u: u * mp.exp(-u))(mp.exp(R._vt(t, y))), pts) / 2
        print(dps, x, mp.nstr(c, 20), mp.nstr(bf, 20), mp.nstr(c / bf - 1, 3), mp.nstr(p / bp - 1, 3),
              stats.landau.cdf(x), stats.landau.pdf(x), round(t1, 2))
