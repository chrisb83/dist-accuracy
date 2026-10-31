import mpmath as mp, vref
s = -1.59e-5; x = mp.mpf(-30.0)
for d in (60, 100, 150, 200, 250):
    mp.mp.dps = d
    a, b, z = vref.params(s)
    C = mp.log(abs(b)) + (a-1)*mp.log(a) - a - mp.loggamma(a)
    f = lambda v: mp.exp(C + (a-1)*mp.log1p(v/b) - a*(v/b))
    q, e = mp.quad(f, mp.linspace(x - 20, x, 41), error=True)
    print(d, mp.nstr(q, 30), mp.nstr(e, 3), mp.nstr(C, 20))
