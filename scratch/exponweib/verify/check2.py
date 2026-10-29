import mpmath as mp
from vfun import g
for d in (100, 200):
    mp.mp.dps = d
    x, a, c = mp.mpf(720.0000000000001), mp.mpf(1e6), mp.mpf(1)
    t = x**c
    cf = -mp.expm1(a*mp.log1p(-mp.exp(-t)))
    # binomial series: sf = sum_{k>=1} (-1)^(k+1) C(a,k) e^{-k t}
    ser = mp.nsum(lambda k: (-1)**(k+1)*mp.binomial(a, k)*mp.exp(-k*t), [1, mp.inf])
    v0 = mp.log(t)
    q = mp.quad(lambda v: g(v, a), mp.linspace(v0, mp.log(t + 600), 200))
    print(d, mp.nstr(cf, 20), mp.nstr(ser, 20), mp.nstr(q, 20))
    x, a, c = mp.mpf(1e-20), mp.mpf(0.3), mp.mpf(50)
    t = x**c
    print('  F5 cdf', mp.nstr((-mp.expm1(-t))**a, 20), mp.nstr(mp.exp(a*c*mp.log(x)), 20))
