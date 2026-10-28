from mpmath import mp

def pdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return c * x**(c-1) / (1 + x**c)**2

def cdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return x**c / (1 + x**c)

def sf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return mp.mpf(1) / (1 + x**c)

def logsf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return -mp.log1p(x**c)

def logcdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return c*mp.log(x) - mp.log1p(x**c)

def logpdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return mp.log(c) + (c-1)*mp.log(x) - 2*mp.log1p(x**c)

def cdf_quad(x, c):
    # independent route: integrate pdf in log variable t = log(x)
    c = mp.mpf(c)
    T = mp.log(mp.mpf(x))
    f = lambda t: c*mp.exp(c*t)/(1+mp.exp(c*t))**2
    return mp.quad(f, [-mp.inf, T])

def ppf(q, c):
    q = mp.mpf(q); c = mp.mpf(c)
    return (q/(1-q))**(1/c)

def isf(q, c):
    q = mp.mpf(q); c = mp.mpf(c)
    return ((1-q)/q)**(1/c)
