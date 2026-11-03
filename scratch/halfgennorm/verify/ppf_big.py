# ppf(0.9, 1e300) by bisection in x on (0.45, 1) with the Kummer-form cdf; check cdf(x_ref) == q.
import mpmath as mp
from scipy import stats
def cdf(x, b):
    a = 1/b; y = x**b
    return x*mp.exp(-y)*mp.hyp1f1(1, a+1, y)/mp.gamma(a+1)
res = {}
for d in (100, 200):
    mp.mp.dps = d
    b = mp.mpf(1e300); q = mp.mpf(0.9)
    lo, hi = mp.mpf('0.45'), mp.mpf('0.999999')
    for _ in range(mp.mp.prec + 10):
        mid = (lo + hi)/2
        if cdf(mid, b) < q: lo = mid
        else: hi = mid
    res[d] = (lo + hi)/2
    print(d, mp.nstr(res[d], 40), 'relchk', mp.nstr(abs(cdf(res[d], b) - q)/q, 3))
mp.mp.dps = 200
print('agree30', abs(res[100]-res[200]) < mp.mpf(10)**-30, 'float', float(res[200]))
print('scipy', stats.halfgennorm.ppf(0.9, 1e300))
