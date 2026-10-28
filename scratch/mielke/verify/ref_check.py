"""Independent mpmath reference for scipy.stats.mielke.

Derived from the docstring pdf  f(x,k,s) = k x^(k-1) / (1+x^s)^(1+k/s).
cdf obtained by antiderivative verification: d/dx [ (1+x^-s)^(-k/s) ] = f(x).
Cross-checked against mp.quad of the pdf at selected points.
"""
import sys
from mpmath import mp

def pdf(x, k, s):
    x, k, s = mp.mpf(x), mp.mpf(k), mp.mpf(s)
    return k * x**(k-1) / (1 + x**s)**(1 + k/s)

def logpdf(x, k, s):
    x, k, s = mp.mpf(x), mp.mpf(k), mp.mpf(s)
    if x == 0:
        # limit: log k + (k-1) log x ; finite only for k==1
        if k == 1:
            return mp.mpf(0)
        return mp.inf if k < 1 else -mp.inf
    return mp.log(k) + (k-1)*mp.log(x) - (1 + k/s)*mp.log1p(x**s)

def cdf(x, k, s):
    # (1 + x^-s)^(-k/s)
    x, k, s = mp.mpf(x), mp.mpf(k), mp.mpf(s)
    return (1 + x**(-s))**(-k/s)

def sf(x, k, s):
    return 1 - cdf(x, k, s)

def ppf(q, k, s):
    # invert cdf: x = (q^(-s/k) - 1)^(-1/s)
    q, k, s = mp.mpf(q), mp.mpf(k), mp.mpf(s)
    return (q**(-s/k) - 1)**(-1/s)

def cdf_quad(x, k, s):
    """Independent route: numerical integration of the pdf."""
    x, k, s = mp.mpf(x), mp.mpf(k), mp.mpf(s)
    # integrate in u = log t to tame the range
    f = lambda u: pdf(mp.e**u, k, s) * mp.e**u
    lo = mp.log(x) - 400/min(k, mp.mpf(1))*mp.log(10)
    return mp.quad(f, [lo, mp.log(x)-30, mp.log(x)])
