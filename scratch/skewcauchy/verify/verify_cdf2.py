# cdf for x<0 via substitution t = -1/u: F(x) = int_0^{-1/x} (1-a)^2 / (pi((1-a)^2 u^2 + 1)) du
import mpmath as mp
def F(x, a):
    x = mp.mpf(x); a = mp.mpf(a); c = (1 - a)**2
    return mp.quad(lambda u: c / (mp.pi * (c * u * u + 1)), [0, -1 / x])
def F_acot(x, a):
    x = mp.mpf(x); a = mp.mpf(a)
    return (1 - a) / mp.pi * mp.acot(-x / (1 - a))
for x, a in [(-1e300, .5), (-1e17, .5), (-1e8, .5), (-1000, .9), (-10, 0.999999999999999), (-1, 0.999999999999999), (-1, 0.99999999)]:
    mp.mp.dps = 100; r1 = F(x, a)
    mp.mp.dps = 200; r2 = F(x, a); r3 = F_acot(x, a)
    print(x, a, mp.nstr(r2, 17), 'p100v200', mp.nstr(abs(r1-r2)/r2, 3), 'quad_vs_acot', mp.nstr(abs(r2-r3)/r3, 3))
# ppf(1e-300,0.5) cdf check with the substitution cdf
mp.mp.dps = 100
x = -(1 - mp.mpf(.5)) / mp.tan(mp.pi * mp.mpf(1e-300) / (1 - mp.mpf(.5)))
print('ppf(1e-300,.5) check', mp.nstr(abs(F(x, .5) - mp.mpf(1e-300)) / mp.mpf(1e-300), 3))
# rescaled substitution integral over [0,1] (avoids quad trouble on a 1e-300 interval)
def G(x, a):
    x = mp.mpf(x); a = mp.mpf(a); c = (1 - a)**2; h = -1 / x
    return h * mp.quad(lambda v: c / (mp.pi * (c * h * h * v * v + 1)), [0, 1])
for dps in (100, 200):
    mp.mp.dps = dps
    print(dps, 'cdf(-1e300,.5)=', mp.nstr(G(-1e300, .5), 30))
    x = -(1 - mp.mpf(.5)) / mp.tan(mp.pi * mp.mpf(1e-300) / (1 - mp.mpf(.5)))
    print(dps, 'ppf(1e-300,.5)=', mp.nstr(x, 30), 'check', mp.nstr(abs(G(x, .5) - mp.mpf(1e-300)) / mp.mpf(1e-300), 3))
