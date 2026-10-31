# F9 sf via Legendre continued fraction for Gamma(a, y) (independent of quad / Kummer)
import mpmath as mp
from scipy import stats
def sf_cf(x, nu, N):
    x = mp.mpf(x); a = mp.mpf(nu); y = a * x * x
    t = mp.mpf(0)
    for n in range(N, 0, -1):
        t = n * (n - a) / (y + 2 * n + 1 - a - t)
    cf = 1 / (y + 1 - a - t)
    return mp.exp(a * mp.log(y) - y - mp.loggamma(a)) * cf
x, nu = 1.3959524698393257, 1000
s = float(stats.nakagami.sf(x, nu))
for d in (100, 200):
    mp.mp.dps = d
    r1, r2 = sf_cf(x, nu, 3000), sf_cf(x, nu, 6000)
    print(d, mp.nstr(r2, 20), 'N-conv', mp.nstr(abs(r1 / r2 - 1), 3), 'ulp', float(abs(mp.mpf(s) - r2) / r2 / mp.mpf(2) ** -52))
