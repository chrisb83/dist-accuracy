# Second route: cdf = Phi(x-c) - Phi(-x-c) with Phi(z) = erfc(-z/sqrt2)/2; plus quad with finer splitting
import mpmath as mp
def Phi(z): return mp.erfc(-z/mp.sqrt(2))/2
def cdf_erfc(x, c):
    x = mp.mpf(x); c = mp.mpf(c); return Phi(x-c) - Phi(-x-c)
def pdf(t, c): return mp.sqrt(2/mp.pi)*mp.cosh(c*t)*mp.exp(-(t*t+c*c)/2)
def cdf_quad(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    pts = sorted(set([mp.mpf(0)] + [x - k for k in (40,30,20,15,10,7,5,3,2,1,0.5) if x-k > 0] + [x]))
    return mp.quad(lambda t: pdf(t, c), pts)
for x, c in [(1.0, 30.0), (965.0, 1000.0)]:
    for dps in (100, 200):
        mp.mp.dps = dps
        print(x, c, dps, mp.nstr(cdf_erfc(x, c), 35), mp.nstr(cdf_quad(x, c), 35))
