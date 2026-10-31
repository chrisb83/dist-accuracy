import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/t/verify')
import mpmath as mp
import vref as V
df274 = 2.7433514990818093
# df = 1: Cauchy closed form (different route)
for dps in (100, 200):
    mp.mp.dps = dps
    print(dps, 'cdf(-1.4e154,1)', mp.nstr(mp.atan(1/mp.mpf(1.4e154))/mp.pi, 35))
    print(dps, 'sf(1.3513e154,1)', mp.nstr(mp.atan(1/mp.mpf(1.351287904186225e154))/mp.pi, 35))
    print(dps, 'cdf(-1e-8,1)', mp.nstr(mp.mpf(1)/2 + mp.atan(mp.mpf(-1e-8))/mp.pi, 35))
# finer quad for tails: substitute u = s - s0 with many breakpoints
def tail(a, df):
    s0 = mp.log(a); df = mp.mpf(df)
    g = lambda s: mp.exp(V.logpdf(mp.exp(s), df) + s)
    pts = [s0 + k for k in [0, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]] + [mp.inf]
    return mp.quad(g, pts)
for q, m in [(1e-308, 'ppf'), (5.6234132519034905e-226, 'ppf'), (3.162277660168379e-151, 'ppf')]:
    out = []
    for dps in (100, 200):
        mp.mp.dps = dps
        x = V.ppf(q, df274)
        out.append((x, tail(-x, df274), V.cdf_beta(x, df274)))
    (x1, t1, b1), (x2, t2, b2) = out
    q = mp.mpf(q)
    print(q, mp.nstr(x2, 20), 'x agree', mp.nstr(abs(x1-x2)/abs(x2), 3),
          'quadcdf rel', mp.nstr(abs(t2-q)/q, 3), 'quad100 vs 200', mp.nstr(abs(t1-t2)/t2, 3),
          'betacdf rel', mp.nstr(abs(b2-q)/q, 3))
