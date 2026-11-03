import mpmath as mp

def logpdf(x, p, a, b):
    x, p, a, b = mp.mpf(x), mp.mpf(p), mp.mpf(a), mp.mpf(b)
    g2 = (a + b) * (a - b)
    s = mp.sqrt(1 + x * x)
    if g2 == 0:
        # p < 0: g^p / K_p(g) -> 2^(1+p) / Gamma(-p)
        lognorm = (1 + p) * mp.log(2) - mp.loggamma(-p)
    else:
        g = mp.sqrt(g2)
        lognorm = p * mp.log(g) - mp.log(mp.besselk(p, g))
    return (lognorm - mp.log(mp.sqrt(2 * mp.pi)) - (p - mp.mpf(1)/2) * mp.log(a)
            + b * x + mp.log(mp.besselk(p - mp.mpf(1)/2, a * s))
            + (p - mp.mpf(1)/2) * mp.log(s))

def pdf(x, p, a, b):
    return mp.exp(logpdf(x, p, a, b))

BP = [mp.mpf(0)] + [mp.mpf(10)**k for k in range(-8, 17)] + [mp.inf]

def tail(x, p, a, b, direction):
    """direction=-1: int_{-inf}^x ; +1: int_x^inf. Rescaled by pdf(x)."""
    x = mp.mpf(x)
    l0 = logpdf(x, p, a, b)
    f = lambda t: mp.exp(logpdf(x + direction * t, p, a, b) - l0)
    tot = mp.mpf(0)
    for lo, hi in zip(BP[:-1], BP[1:]):
        tot += mp.quad(f, [lo, hi])
    return tot * mp.exp(l0)

def cdf(x, p, a, b): return tail(x, p, a, b, -1)
def sf(x, p, a, b): return tail(x, p, a, b, +1)

def mid(x0, x1, p, a, b):
    """int_{x0}^{x1} pdf, finite interval (for symmetric b=0 cases)."""
    return mp.quad(lambda t: pdf(t, p, a, b), mp.linspace(mp.mpf(x0), mp.mpf(x1), 9))
