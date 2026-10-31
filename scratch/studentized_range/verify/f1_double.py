# F1 (1 < k < 2): double-precision reference (the 2-D mp.quad was too slow).
# log-space integrand, D from the far tail via log_ndtr; outer over u = log s.
# Two independent quadratures (adaptive quad, tanh-sinh-free; and dense trapezoid) must agree.
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy import integrate, special as sc, stats
LN2PI = 0.5*np.log(2*np.pi)
def logD(z, w):
    if w < 1e-6:   # D = w*phi(z+w/2)*(1+O(w^2))
        return np.log(w) - (z+w/2)**2/2 - LN2PI
    if z + w/2 > 0:
        a, b = sc.log_ndtr(-z), sc.log_ndtr(-z-w)
    else:
        a, b = sc.log_ndtr(z+w), sc.log_ndtr(z)
    return a + np.log(-np.expm1(b - a))
def lr(z, w, k):
    return np.log(k*(k-1)) - z*z/2 - (z+w)**2/2 - 2*LN2PI + (k-2)*logD(z, w)
def logfS(s, df):
    return (df/2)*np.log(df) + (df-1)*np.log(s) - df*s*s/2 - (df/2-1)*np.log(2) - sc.gammaln(df/2)
def rk_quad(w, k):
    c = -w/2
    return integrate.quad(lambda z: np.exp(lr(z, w, k)), -np.inf, np.inf, points=None, epsabs=0, epsrel=1e-13, limit=500)[0] if False else \
        sum(integrate.quad(lambda z: np.exp(lr(z, w, k)), a, b, epsabs=0, epsrel=1e-13, limit=500)[0]
            for a, b in [(c-40, c-6), (c-6, c), (c, c+6), (c+6, c+40)])
def rk_trap(w, k, h=0.01):
    z = np.arange(-w/2-40, -w/2+40, h)
    return h*sum(np.exp(lr(zi, w, k)) for zi in z)
def pdf(x, k, df, rk):
    sd = 1/np.sqrt(2*df)
    if df > 1000:
        edges = np.arange(-12, 12.01, 1)*sd
    else:
        edges = np.array([-40, -20, -10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3.0])
    g = lambda u: np.exp(logfS(np.exp(u), df) + 2*u)*rk(x*np.exp(u), k)
    return sum(integrate.quad(g, a, b, epsabs=0, epsrel=1e-12, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
for x, k, df in [(1.0, 1.5, 10), (3.0, 1.9, 10), (1.0, 1.5, 1e6)]:
    v1 = pdf(x, k, df, rk_quad)
    v2 = pdf(x, k, df, lambda w, k: rk_trap(w, k, 0.02))
    with np.errstate(all='ignore'):
        sv = stats.studentized_range.pdf(x, k, df)
    print(f"pdf({x}, {k}, {df}) scipy={sv!r} ref_quad={v1!r} ref_trapz={v2!r} rel.diff={abs(v1-v2)/v1:.2e}", flush=True)
