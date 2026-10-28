import mpmath as mp
x0 = 14120.906486483083; df = 2e8
for dps in (100, 200):
    mp.mp.dps = dps
    x = mp.mpf(x0); k2 = mp.mpf(df)/2
    c = mp.log(2) - (k2-1)*mp.log(2) - mp.loggamma(k2)
    lx = (df-1)*mp.log(x) - x*x/2 + c   # log pdf at x
    # pdf(x-u)/pdf(x) = exp((df-1)*log1p(-u/x) + x*u - u^2/2)
    g = lambda u: mp.exp((df-1)*mp.log1p(-u/x) + x*u - u*u/2)
    pts = [0, mp.mpf('0.003'), mp.mpf('0.01'), mp.mpf('0.03'), mp.mpf('0.1'), mp.mpf('0.3'), 1, 3, 10, 30]
    I, err = mp.quad(g, pts, error=True, maxdegree=12)
    print(dps, mp.nstr(mp.exp(lx)*I, 35), mp.nstr(err/I, 3))
