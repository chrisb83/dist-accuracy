# independent quadrature check of F6 cdf values (integral of the docstring pdf in t = log x)
import mpmath as mp
def logPhi(w):
    if w > 0: return mp.log1p(-mp.erfc(w/mp.sqrt(2))/2)
    return mp.log(mp.erfc(-w/mp.sqrt(2))/2)
def cdf(x,c,s): return -mp.expm1(mp.mpf(c)*logPhi(-mp.log(mp.mpf(x))/mp.mpf(s)))
for dps in (40, 60):
    mp.mp.dps = dps
    for x, c, s in [(2.104223637677623e-17,1e20,1),(3.1391327920480296e-17,1e10,1)]:
        c, s = mp.mpf(c), mp.mpf(s)
        f = lambda t: c/s*mp.npdf(t/s)*mp.exp((c-1)*logPhi(-t/s))
        T = mp.log(mp.mpf(x))
        pts = [T-4] + [T - mp.mpf(k)/200 for k in range(400, -1, -1)]
        print(dps, mp.nstr(mp.quad(f, pts), 20), mp.nstr(cdf(x, c, s), 20))
