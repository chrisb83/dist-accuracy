import mpmath as mp, numpy as np, scipy, scipy.stats as st
print(scipy.__version__)

def Phi_c(z):  # complementary route: 1 - Q(z), Q via erfc
    return 1 - mp.erfc(z/mp.sqrt(2))/2

def logpdf(x, a):
    x = mp.mpf(x); a = mp.mpf(a)
    z = a - 1/x
    # log of docstring pdf, with Phi(a) via quad of the normal density
    Phia = mp.quad(lambda t: mp.exp(-t**2/2), [-mp.inf, 0, min(a, mp.mpf(60))])/mp.sqrt(2*mp.pi)
    return -2*mp.log(x) - mp.log(Phia) - mp.log(mp.sqrt(2*mp.pi)) - z**2/2

def cdf(x, a):
    x = mp.mpf(x); a = mp.mpf(a)
    return Phi_c(a - 1/x)/Phi_c(a)

cases = [('logpdf', 5.3e-155, 3.570477051665046), ('logpdf', 6e-155, 3.570477051665046),
         ('logpdf', 7.4e-155, 3.570477051665046), ('logpdf', 5.3e-155, 0.001),
         ('logpdf', 7e-155, 0.001), ('logpdf', 5.5e-155, 0.5),
         ('cdf', 1e-154, 1e154), ('logpdf', 1e-154, 1e154)]
f = {'logpdf': logpdf, 'cdf': cdf}
for m, x, a in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(f[m](x, a))
    mp.mp.dps = 200
    agree = vals[0] == 0 or abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    s = getattr(st.alpha, m)(x, a)
    ref = float(vals[1])
    err = abs(s-ref)/abs(ref)/2.22e-16 if np.isfinite(s) else float('inf')
    print(m, x, a, 'scipy', repr(s), 'ref', mp.nstr(vals[1], 16), 'agree30', agree, 'ulp', err)
mp.mp.dps = 50
print('z exact for F2:', mp.nstr(mp.mpf(1e154) - 1/mp.mpf(1e-154), 6), ' 1/x rounded == a:', 1.0/1e-154 == 1e154)
