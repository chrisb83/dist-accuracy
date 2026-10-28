"""Independent verifier reference for chi2 (standard form).
pdf from docstring: f(x,k) = x^(k/2-1) exp(-x/2) / (2^(k/2) Gamma(k/2)).
cdf/sf: quadrature of the pdf (mp.quad), scaled by f at the endpoint, on
geometric intervals away from the endpoint. Tiny df: alternating series
P(a,y) = y^a/Gamma(a) * sum (-y)^k/(k!(a+k)). Quantiles: Newton on these.
"""
import sys, mpmath as mp

def logpdf(x, k):
    x = mp.mpf(x); k = mp.mpf(k)
    return (k/2-1)*mp.log(x) - x/2 - (k/2)*mp.log(2) - mp.loggamma(k/2)

def pdf(x, k):
    return mp.exp(logpdf(x, k))

def _tailint(x, k, direction):
    """int of f from x towards 0 (direction=-1) or inf (+1), = f(x)*I."""
    x = mp.mpf(x); k = mp.mpf(k)
    L0 = logpdf(x, k)
    d = abs((k/2-1)/x - mp.mpf(1)/2)
    s = min(1/d if d > 0 else x, mp.sqrt(2*k) if k > 2 else x, x if direction < 0 else mp.inf)
    s = max(s, mp.mpf(1)) if direction > 0 else s
    g = lambda t: mp.exp(logpdf(t, k) - L0)
    cut = -(mp.mp.dps*mp.log(10) + 50)
    pts = [x]; w = s/4
    while True:
        nxt = pts[-1] + direction*w
        if direction < 0 and nxt <= 0:
            pts.append(mp.mpf(0)); break
        pts.append(nxt)
        if logpdf(nxt, k) - L0 < cut: break
        w *= 2
    tot = 0
    for a, b in zip(pts[:-1], pts[1:]):
        tot += mp.quad(g, sorted([a, b]))
    return mp.exp(L0)*tot

def cdf(x, k): return _tailint(x, k, -1)
def sf(x, k):  return _tailint(x, k, +1)

def cdf_small(x, k):
    a = mp.mpf(k)/2; y = mp.mpf(x)/2
    s = 0; t = mp.mpf(1); n = 0
    while True:
        term = t/(a+n)
        s += term
        if abs(term) < mp.eps*abs(s)*1e-5 and n > 2: break
        n += 1; t *= -y/n
    return y**a/mp.gamma(a)*s

def newton(F, x0, q, k):
    x = mp.mpf(x0)
    for _ in range(60):
        dx = (F(x, k) - q)/pdf(x, k)
        x -= dx
        if abs(dx) < abs(x)*mp.mpf(10)**(-mp.mp.dps+5): break
    return x

def newton_log(F, x0, q, k):  # Newton on log F for tiny-x quantiles
    x = mp.mpf(x0)
    for _ in range(200):
        Fx = F(x, k)
        dx = (mp.log(Fx) - mp.log(q))*Fx/pdf(x, k)
        x -= dx
        if abs(dx) < abs(x)*mp.mpf(10)**(-mp.mp.dps+5): break
    return x
