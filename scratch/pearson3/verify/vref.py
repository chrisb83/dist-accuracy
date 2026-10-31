# Verifier reference for pearson3: pdf from docstring, cdf/sf by mp.quad of the pdf in x,
# ppf by root-finding on the quadrature cdf. Shapes are the exact double values.
import mpmath as mp

def params(skew):
    s = mp.mpf(skew)
    beta = 2 / s
    alpha = beta**2
    zeta = -beta
    return alpha, beta, zeta

def logpdf(x, skew):
    alpha, beta, zeta = params(skew)
    t = beta * (mp.mpf(x) - zeta)
    if t <= 0:
        return mp.ninf
    return mp.log(abs(beta)) + (alpha - 1) * mp.log(t) - t - mp.loggamma(alpha)

def pdf(x, skew):
    return mp.exp(logpdf(x, skew))

OFFS = [80, 50, 30, 20, 12, 8, 5, 3, 2, 1, 0.5, 0.25, 0.1, 0]

def _left(x, skew):
    # integral of pdf from -inf (or support edge) to x
    alpha, beta, zeta = params(skew)
    x = mp.mpf(x)
    pts = [x - o for o in OFFS]
    if beta > 0:  # support starts at zeta
        if x <= zeta:
            return mp.mpf(0)
        pts = [p for p in pts if p > zeta]
        if pts[0] > zeta and x - 80 <= zeta:
            pts = [zeta] + pts
    L0 = logpdf(x, skew)  # rescale: mp.quad uses an absolute tolerance
    return mp.exp(L0) * mp.quad(lambda u: mp.exp(logpdf(u, skew) - L0), pts)

def _right(x, skew):
    alpha, beta, zeta = params(skew)
    x = mp.mpf(x)
    pts = [x + o for o in reversed(OFFS)]
    if beta < 0:
        if x >= zeta:
            return mp.mpf(0)
        pts = [p for p in pts if p < zeta]
        if x + 80 >= zeta:
            pts = pts + [zeta]
    L0 = logpdf(x, skew)
    return mp.exp(L0) * mp.quad(lambda u: mp.exp(logpdf(u, skew) - L0), pts)

def cdf(x, skew):
    return _left(x, skew)

def sf(x, skew):
    return _right(x, skew)

def ppf(q, skew, x0):
    q = mp.mpf(q)
    if q < 0.5:
        f = lambda x: mp.log(_left(x, skew)) - mp.log(q)
    else:
        f = lambda x: mp.log(_right(x, skew)) - mp.log(1 - q)
    # Newton with derivative from pdf
    x = mp.mpf(x0)
    for _ in range(60):
        if q < 0.5:
            F = _left(x, skew); step = (mp.log(F) - mp.log(q)) * F / pdf(x, skew)
        else:
            S = _right(x, skew); step = -(mp.log(S) - mp.log(1 - q)) * S / pdf(x, skew)
        x -= step
        if abs(step) < mp.mpf(10)**(-mp.mp.dps + 10) * max(1, abs(x)):
            break
    return x

def sf_edge(x, skew):
    alpha, beta, zeta = params(skew)
    if beta > 0 and mp.mpf(x) <= zeta:
        return mp.mpf(1)
    return sf(x, skew)

def cdf_edge(x, skew):
    alpha, beta, zeta = params(skew)
    if beta < 0 and mp.mpf(x) >= zeta:
        return mp.mpf(1)
    return cdf(x, skew)
