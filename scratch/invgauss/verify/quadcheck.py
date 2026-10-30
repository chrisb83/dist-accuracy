import mpmath as mp
from verify import sf, pdf, cdf
def sf_quad(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    # right-tail decay rate of log pdf at x: d/dx[-(x-mu)^2/(2mu^2 x)] - 1.5/x
    r = (x*x - mu*mu)/(2*mu*mu*x*x) + mp.mpf(3)/(2*x)
    h = 1/r
    g = lambda s: pdf(x + s*h, mu)*h
    v, e = mp.quad(g, [0, 1, 4, 16, 64, 256, mp.inf], error=True, maxdegree=10)
    return v
for d in (50, 80):
    mp.mp.dps = d
    for a, s in [(1e9,1e3),(1e3,1.0),(1e19,1e8),(0.002,0.001),(1e40,1e20)]:
        v1 = sf(a, s); v2 = sf_quad(a, s)
        print(d, a, s, mp.nstr(v1, 25), mp.nstr(v2, 25), mp.nstr(abs(v1-v2)/v1, 3))
mp.mp.dps = 100
# logcdf with log1p(-sf) where sf small
for a, s in [(1e9,1e3),(1e40,1e20)]:
    print('logcdf', a, s, mp.nstr(mp.log1p(-sf(a, s)), 17))
    mp.mp.dps=200; v=mp.log1p(-sf(a,s)); mp.mp.dps=100; print('  200:', mp.nstr(v,17))
