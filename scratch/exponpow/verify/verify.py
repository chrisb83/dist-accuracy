import mpmath as mp, numpy as np
from scipy import stats
def logpdf(x, b):
    x, b = mp.mpf(x), mp.mpf(b)
    xb = x**b
    # route: log of pdf product, with exp(xb)-1-xb via series-free high precision
    # log(b) + (b-1) log x - sum_{k>=2} xb^k/k!  (series, no cancellation)
    ser = mp.nsum(lambda k: xb**k/mp.factorial(k), [2, mp.inf])
    return mp.log(b) + (b-1)*mp.log(x) - ser
def ppf(q, b):
    q, b = mp.mpf(q), mp.mpf(b)
    return mp.log(1 - mp.log(1 - q))**(1/b)
def isf(q, b):
    q, b = mp.mpf(q), mp.mpf(b)
    return mp.log(1 - mp.log(q))**(1/b)
def cdf_quad(x, b):
    # integral of pdf, substitution t = x**b: int_0^{x^b} exp(1+t-e^t) dt
    T = mp.mpf(x)**mp.mpf(b)
    return mp.quad(lambda t: mp.exp(1 + t - mp.exp(t)), [0, T])
cases = [('logpdf',1e-10,1.0),('logpdf',1e-150,1.0),('logpdf',1e-5,1.0),('logpdf',1e-3,1.0),
         ('logpdf',1e-10,1.000000000001),('logpdf',1.0000001151292594e-05,1.00000001),
         ('logpdf',1e-10,1.00000001),
         ('ppf',0.49999999999999994,0.001),('isf',0.15,0.001)]
for meth,a,b in cases:
    f = {'logpdf':logpdf,'ppf':ppf,'isf':isf}[meth]
    vals=[]
    for d in (100,200):
        mp.mp.dps=d; vals.append(f(a,b))
    mp.mp.dps=200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    s = getattr(stats.exponpow, meth)(a, b)
    ref = vals[1]
    ulp = float(abs(mp.mpf(s)-ref)/abs(ref)/mp.mpf(2.220446049250313e-16))
    extra=''
    if meth in ('ppf','isf'):
        mp.mp.dps=50
        c = cdf_quad(ref, b)
        tgt = a if meth=='ppf' else 1-mp.mpf(a)
        extra=f' quadcheck relerr={float(abs(c-tgt)/tgt):.1e}'
    print(meth,a,b,'scipy',repr(s),'ref',mp.nstr(ref,17),'ulp %.3g'%ulp,'agree',agree,extra)
