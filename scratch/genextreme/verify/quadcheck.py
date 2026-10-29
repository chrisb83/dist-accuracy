# Second route: cdf/sf by numerical integration of the docstring pdf, in u = 1 - c*x
import mpmath as mp
def pdf_u(u, c):  # pdf as function of u = 1 - c x; dx = -du/c
    return mp.exp(-mp.power(u, 1/c)) * mp.power(u, 1/c - 1)
for dps in (100, 200):
    mp.mp.dps = dps
    c = mp.mpf(1000); x = mp.mpf(-1e308)
    u0 = 1 - c*x
    # cdf(x) = int_{-inf}^x pdf dx = (1/c) int_{u0}^{inf} pdf_u du; substitute u = exp(s)
    g = lambda s: pdf_u(mp.exp(s), c) * mp.exp(s) / c
    s0 = mp.log(u0)
    cdf1 = mp.quad(g, [s0] + [s0 + k*1000 for k in (1,2,4,8,16,32,64,128,256,512,800)])
    c2 = mp.mpf(0.7); x2 = mp.mpf(1.4285714285714286)
    u2 = 1 - c2*x2
    sf2 = mp.quad(lambda u: pdf_u(u, c2) / c2, [0, u2])
    print(dps, mp.nstr(cdf1, 35), mp.nstr(sf2, 35))
