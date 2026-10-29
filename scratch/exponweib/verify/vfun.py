import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter('ignore')
ew = stats.exponweib

# dF/dv in v = log(s), s = x**c:  a (1-e^-s)^(a-1) e^-s s
def g(v, a):
    s = mp.exp(v)
    return a * (-mp.expm1(-s))**(a-1) * mp.exp(-s) * s

def cdf_q(x, a, c):
    v0 = c*mp.log(x)
    pts = [-mp.inf] + [v0 - k for k in (200, 50, 10, 2) ] + [v0]
    return mp.quad(lambda v: g(v, a), pts)

def sf_q(x, a, c):
    v0 = c*mp.log(x)
    top = max(v0, mp.mpf(0)) + 8
    pts = sorted(set([v0, min(v0+2, top), mp.log(mp.log(a)+1) if a > 3 and mp.log(mp.log(a)+1) > v0 else v0, top]))
    return mp.quad(lambda v: g(v, a), pts) + mp.quad(lambda v: g(v, a), [top, mp.inf])

def logpdf_m(x, a, c):
    t = x**c
    return mp.log(a)+mp.log(c)+(a-1)*mp.log(-mp.expm1(-t)) - t + (c-1)*mp.log(x)

def ppf_m(q, a, c):  # closed form, then checked by quad cdf
    return (-mp.log(1 - q**(1/a)))**(1/c)

def isf_m(p, a, c):
    return ppf_m(1 - p, a, c)

