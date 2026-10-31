# Third route: closed form via erfc at 400 and 600 dps; plus the analyst's ref for comparison.
import sys, mpmath as mp
sys.path.insert(0, 'refs'); sys.path.insert(0, '.claude/skills/dist-accuracy-review/scripts')
def cdf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu); r = mp.sqrt(2*x)
    return mp.erfc((1/mu - x)/r)/2 - mp.exp(2/mu)*mp.erfc((1/mu + x)/r)/2
def sf(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu); r = mp.sqrt(2*x)
    return mp.erfc((x - 1/mu)/r)/2 + mp.exp(2/mu)*mp.erfc((1/mu + x)/r)/2
import recipinvgauss_ref as R
for m, x, mu in [('cdf',500,0.001),('cdf',79.36507936507935,0.003),('cdf',1e-19,1e8),('cdf',1e-9,1000),
                 ('cdf',0.07142857142857142,0.1),('cdf',6.5,0.01),('sf',1400,0.1),('sf',1400,0.01)]:
    f = cdf if m == 'cdf' else sf
    v = []
    for d in (400, 600):
        mp.mp.dps = d; v.append(f(x, mu))
    mp.mp.dps = 400
    a = getattr(R, m)(mp.mpf(x), mp.mpf(mu))
    print(m, x, mu, mp.nstr(v[1], 20), 'stable' if abs(v[0]-v[1]) < abs(v[1])*mp.mpf(10)**-40 else 'UNSTABLE', 'analyst', mp.nstr(a, 20))
