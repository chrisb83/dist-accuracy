import mpmath as mp, sys
sys.path.insert(0, '.')
from verify import pdf, cdf_closed, isf, ppf
def cdf_quad2(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    pts = sorted({x - m*mp.mpf(10)**k for k in range(-4, 5) for m in range(1, 10)})
    return mp.quad(lambda y: pdf(y, c), [-mp.inf] + pts + [x], maxdegree=10)
def isf2(q, c):
    q, c = mp.mpf(q), mp.mpf(c)
    return -mp.log(mp.expm1(-mp.log1p(-q)/c))
for d in (100, 200):
    mp.mp.dps = d
    for x, c in [(-1000, 0.41192440799679475), (8, 1e6), (-710, 1), (-1000, 0.41192440799679475)]:
        a, b = cdf_quad2(x, c), cdf_closed(x, c)
        print(d, x, c, mp.nstr(b, 20), mp.nstr(abs(a/b-1), 3))
    r = isf2(1e-300, 50); print(d, "isf(1e-300,50)", mp.nstr(r, 20), "resid", mp.nstr((1-cdf_closed(r,50))/mp.mpf(1e-300)-1, 3))
from scipy import stats; print(stats.genlogistic.isf(1e-300, 50))
