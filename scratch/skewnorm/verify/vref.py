import mpmath as mp

def pdf(t, a):
    if abs(t) > 1e8 or abs(a*t) > 1e8:
        return mp.mpf(0)
    return 2*mp.npdf(t)*mp.ncdf(a*t)

def cdf(x, a):
    # direct quadrature of the docstring pdf, valid for x <= 0 small side
    x = mp.mpf(x); a = mp.mpf(a)
    s = 1/((1+a*a)*max(abs(x), 1/mp.sqrt(1+a*a)))
    pts = [x - k*s for k in (256, 128, 64, 32, 16, 8, 4, 2, 1)] + [x]
    return mp.quad(lambda t: pdf(t, a), [-mp.inf] + pts)

def sf(x, a):
    return cdf(-mp.mpf(x), -mp.mpf(a))

def logpdf(x, a):
    x = mp.mpf(x); a = mp.mpf(a)
    return mp.log(2) - x*x/2 - mp.log(mp.sqrt(2*mp.pi)) + mp.log(mp.ncdf(a*x))

def ppf(q, a, lo, hi):
    q = mp.mpf(q)
    f = lambda x: mp.log(cdf(x, a)) - mp.log(q)
    return mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver='anderson', tol=mp.mpf(10)**(-mp.mp.dps+10))
