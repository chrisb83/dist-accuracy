# Quad of pdf with dense splitting near the upper limit (integrand ~ exp(-35 u) decay from x downward)
import mpmath as mp
def pdf(t, c): return mp.sqrt(2/mp.pi)*mp.cosh(c*t)*mp.exp(-(t*t+c*c)/2)
for dps in (100, 200):
    mp.mp.dps = dps
    x, c = mp.mpf(965), mp.mpf(1000)
    pts = [mp.mpf(0)] + [x - mp.mpf(k)/8 for k in range(400, 0, -1)] + [x]
    print(dps, mp.nstr(mp.quad(lambda t: pdf(t, c), pts), 35))
