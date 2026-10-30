"""Verifier reference for norminvgauss (standard form).
cdf/sf: normal variance-mean mixture X|V ~ N(bV, V), V ~ IG(mean 1/gamma, shape 1),
integrated with mp.quad in s = log(gamma*V), breakpoints placed around the
maximum of the log-integrand (found by grid + refinement).
pdf: mp.besselk directly from the docstring formula.
"""
import mpmath as mp

def _gam(a, b):
    a = mp.mpf(a); b = mp.mpf(b)
    return mp.sqrt((a + b) * (a - b))

def _logint(s, x, a, b, upper):
    g = _gam(a, b)
    v = mp.exp(s) / g
    z = (mp.mpf(x) - mp.mpf(b) * v) / mp.sqrt(v)
    if upper:
        z = -z
    # log of IG density * v (ds measure); exponent -(gamma v - 1)^2/(2v) written exactly
    lg = -mp.log(2 * mp.pi) / 2 - mp.log(v) / 2 - g * mp.expm1(s) ** 2 / (2 * mp.exp(s))
    if z < -mp.mpf(10) ** 8:
        # far tail only (contribution exp(-5e15) relative to the peak; never decisive)
        lp = -z * z / 2 - mp.log(-z) - mp.log(2 * mp.pi) / 2
    elif z > mp.mpf(10) ** 8:
        lp = mp.mpf(0)  # log(1 - tiny), tiny < exp(-5e15)
    elif z > 50:
        lp = mp.log1p(-mp.erfc(z / mp.sqrt(2)) / 2)
    else:
        lp = mp.log(mp.ncdf(z))
    return lp + lg

def _tail(x, a, b, upper):
    L = lambda s: _logint(s, x, a, b, upper)
    # coarse grid
    best = None
    for k in range(-1000, 1001):
        s = mp.mpf(k)
        val = L(s)
        if best is None or val > best[1]:
            best = (s, val)
    s0 = best[0]
    # golden section refinement in [s0-1, s0+1]
    lo, hi = s0 - 1, s0 + 1
    gr = (mp.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    fc, fd = L(c), L(d)
    for _ in range(2000):
        if hi - lo < mp.mpf(10) ** (-(9 * mp.mp.dps) // 10) * (1 + abs(s0)):
            break
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo); fc = L(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo); fd = L(d)
    sm = (lo + hi) / 2
    Lm = L(sm)
    # half widths where L drops by 1, by bisection in log(d)
    def width(sign):
        dlo, dhi = mp.mpf(10) ** -200, mp.mpf(50)
        if L(sm + sign * dhi) > Lm - 1:
            return dhi
        for _ in range(200):
            dm = mp.sqrt(dlo * dhi)
            if L(sm + sign * dm) > Lm - 1:
                dlo = dm
            else:
                dhi = dm
            if dhi / dlo < 1 + mp.mpf(10) ** -3:
                break
        return dhi
    wl, wr = width(-1), width(+1)
    D = mp.mpf(2.31) * mp.mp.dps + 30   # drop beyond which the rest is < 10**-dps relative
    def edge(sign, w):
        k = mp.mpf(1)
        while L(sm + sign * k * w) > Lm - D:
            k *= 2
        return k
    kl, kr = edge(-1, wl), edge(+1, wr)
    pts = []
    k = kl
    while k >= 1:
        pts.append(sm - k * wl); k /= 2
    pts.append(sm)
    k = mp.mpf(1)
    while k <= kr:
        pts.append(sm + k * wr); k *= 2
    f = lambda s: mp.exp(L(s) - Lm)
    return mp.quad(f, pts) * mp.exp(Lm)

def cdf(x, a, b):
    return _tail(x, a, b, False)

def sf(x, a, b):
    return _tail(x, a, b, True)

def pdf(x, a, b):
    a = mp.mpf(a); b = mp.mpf(b); x = mp.mpf(x)
    sq = mp.sqrt(1 + x * x)
    return a / mp.pi * mp.besselk(1, a * sq) * mp.exp(_gam(a, b) + b * x) / sq

def isf(q, a, b, x0):
    q = mp.mpf(q)
    lq = mp.log(q)
    x0 = mp.mpf(x0)
    x1 = x0 * (1 + mp.mpf('1e-8')) if x0 != 0 else mp.mpf('1e-20')
    return mp.findroot(lambda t: mp.log(sf(t, a, b)) - lq, (x0, x1), solver='secant',
                       tol=mp.mpf(10) ** (-2 * mp.mp.dps // 3), maxsteps=60)
