# logpdf(0.5, 1e300): direct formula at 400/600 digits (200 digits cannot resolve 1e-300 against 690),
# and independently via the series -lgamma(1+a) = euler*a - sum_{k>=2} zeta(k)(-a)^k/k, a = 1/beta.
import mpmath as mp
from scipy import stats
b = mp.mpf(10)**300  # exact? use float value of 1e300
b = mp.mpf(1e300)
for d in (400, 600):
    mp.mp.dps = d
    a = 1/b
    direct = mp.log(b) - mp.loggamma(a) - mp.mpf(0.5)**b
    series = mp.euler*a - sum(mp.zeta(k)*(-a)**k/k for k in range(2, 6)) - mp.mpf(0.5)**b
    print(d, mp.nstr(direct, 35), mp.nstr(series, 35))
print('scipy', stats.halfgennorm.logpdf(0.5, 1e300))
