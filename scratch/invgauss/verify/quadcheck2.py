import mpmath as mp
from verify import sf, pdf
mp.mp.dps = 40
for a, s in [(1e3,1.0),(1e9,1e3),(0.002,0.001)]:
    x = mp.mpf(a); mu = mp.mpf(s)
    r = (x*x - mu*mu)/(2*mu*mu*x*x) + mp.mpf(3)/(2*x); h = 1/r
    g = lambda t: pdf(x + t*h, mu)*h
    pts = [mp.mpf(k)/4 for k in range(0, 4*120+1)]
    v = mp.fsum(mp.quad(g, [pts[i], pts[i+1]]) for i in range(len(pts)-1))
    v += mp.quad(g, [120, mp.inf])
    ref = sf(a, s)
    print(a, s, mp.nstr(ref, 25), mp.nstr(v, 25), mp.nstr(abs(v-ref)/ref, 3))
