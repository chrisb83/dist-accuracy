import mpmath as mp
def isf2(q, c):
    q, c = mp.mpf(q), mp.mpf(c); return -mp.log(mp.expm1(-mp.log1p(-q)/c))
def sf(x, c):
    x, c = mp.mpf(x), mp.mpf(c); return -mp.expm1(-c*mp.log1p(mp.exp(-x)))
def sf_quad(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    f = lambda y: mp.exp(mp.log(c) - y - (c+1)*mp.log1p(mp.exp(-y)))
    return mp.quad(f, [x, x+1, x+10, x+100, mp.inf])
for d in (100, 200):
    mp.mp.dps = d
    r = isf2(1e-300, 50)
    print(d, mp.nstr(r, 20), "sf resid", mp.nstr(sf(r, 50)/mp.mpf(1e-300)-1, 3), "quad resid", mp.nstr(sf_quad(r, 50)/mp.mpf(1e-300)-1, 3))
for d in (100, 200):
    mp.mp.dps = d
    r = isf2(1e-300, 50); c = mp.mpf(50)
    f = lambda u: mp.exp(mp.log(c) - (r+u) - (c+1)*mp.log1p(mp.exp(-(r+u))))
    print(d, "shifted quad resid", mp.nstr(mp.quad(f, [0, 1, 10, 100, mp.inf])/mp.mpf(1e-300)-1, 3))
for d in (100, 200):
    mp.mp.dps = d
    r = isf2(1e-300, 50); c = mp.mpf(50); q = mp.mpf(1e-300)
    f = lambda u: mp.exp(mp.log(c) - (r+u) - (c+1)*mp.log1p(mp.exp(-(r+u)))) / q
    print(d, "scaled quad resid", mp.nstr(mp.quad(f, [0, 1, 10, 100, mp.inf]) - 1, 3))
