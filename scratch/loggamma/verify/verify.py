# Independent verification of loggamma findings F1-F5 (verifier).
import sys, numpy as np, mpmath as mp
from scipy import stats

def logpdf(x, c): return c*x - mp.exp(x) - mp.loggamma(c)
def pdf(x, c): return mp.exp(c*x - mp.exp(x)) / mp.gamma(c)

def cdf_quad(x, c):
    # integrate the docstring pdf; peaked near log(c) with width 1/sqrt(c)
    w = 1/mp.sqrt(c)
    pts = [x - k*w for k in (200, 60, 30, 15, 8, 4, 2, 1, 0)]
    return mp.quad(lambda t: pdf(t, c), pts)

def sf_upper(x, c):  # complementary function directly
    return mp.gammainc(c, mp.exp(x), mp.inf, regularized=True)

def cdf_series(x, c):  # P(c,z) = z^c e^-z sum z^k / Gamma(c+k+1)
    z = mp.exp(x)
    return z**c*mp.exp(-z)*mp.nsum(lambda k: z**k/mp.gamma(c+k+1), [0, mp.inf])

def run(dps):
    mp.mp.dps = dps
    M = mp.mpf
    out = {}
    c = M('1e-310')
    out['F1 logpdf(0,1e-310)'] = logpdf(M(0), c)
    out['F1 cdf(-1,1e-310)'] = 1 - sf_upper(M(-1), c)
    # ppf(0.99, 1e-310): solve P(c, e^x) = q with P ~ e^{cx}/Gamma(c+1) (+ O(e^x)); verify below
    q = M('0.99')
    xq = (mp.log(q) + mp.loggamma(c+1))/c
    out['F1 ppf(0.99,1e-310)'] = xq
    out['F1 check cdf(xq)-q'] = cdf_series(xq, c) - q
    c6 = M(10)**6
    for x in ['13.810910557964274', '13.809510557964273', '13.805510557964274']:
        out['F2 cdf(%s,1e6)' % x] = cdf_quad(M(float(x)), c6)
    out['F2 sf(13.810910557964274,1e6)'] = 1 - cdf_quad(M(float('13.810910557964274')), c6)
    out['F3 sf(-800,1e-14)'] = sf_upper(M(-800), M('1e-14'))
    out['F3 sf(-800,1e-14) via 1-series'] = 1 - cdf_series(M(-800), M('1e-14'))
    out['F3 sf(-708.3964185322642,1e-10)'] = sf_upper(M(float('-708.3964185322642')), M('1e-10'))
    out['F4 pdf(13.845748741024885,1e6)'] = pdf(M(float('13.845748741024885')), c6)
    out['F4 pdf(13.815210557964274,1e6)'] = pdf(M(float('13.815210557964274')), c6)
    out['F4 logpdf(13.818510557964274,1e6)'] = logpdf(M(float('13.818510557964274')), c6)
    # F5 isf: solve 1 - cdf(x) = q via findroot on the quad cdf? use gammainc lower for root, check with quad
    for qs in ['0.9999993371296838', '1e-08']:
        target = (1 - M(qs)) if qs.startswith('0.9') else M(qs)
        f = lambda x: mp.log(cdf_quad(x, c6)) - mp.log(target)
        xr = mp.findroot(f, M('13.8105'), tol=mp.mpf(10)**(-2*dps+20))
        out['F5 %s(%s,1e6)' % ('isf' if qs.startswith('0.9') else 'ppf', qs)] = xr
        out['F5 check quadcdf/target-1 (%s)' % qs] = cdf_quad(xr, c6)/target - 1
    return out

a = run(100); b = run(200)
for k in a:
    agree = mp.almosteq(a[k], b[k], rel_eps=mp.mpf(10)**-30) if b[k] != 0 else True
    print(k, mp.nstr(b[k], 20), 'agree30' if agree else 'DISAGREE', mp.nstr(a[k], 20) if not agree else '')
