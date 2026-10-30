import mpmath as mp, time, functools

def _logu_theta(t, x):   # theta in (0, pi/2]
    if t == 0:
        return -x - 1
    return mp.log(t / mp.sin(t)) - x - t * mp.cot(t)

def _logu_eps(e, x):     # theta = pi - eps, eps in (0, pi/2]
    return mp.log((mp.pi - e) / mp.sin(e)) - x + (mp.pi - e) * mp.cot(e)

def _solve(fn, L, a, b, inc):
    # log u = L on [a, b]; fn increasing (inc) or decreasing in the variable
    for _ in range(200):
        m = (a + b) / 2 if b / a < 4 else mp.sqrt(a * b)
        v = fn(m)
        if (v < L) == inc:
            a = m
        else:
            b = m
        if b - a < 1e-6 * b:
            break
    return (a + b) / 2

@functools.lru_cache(maxsize=4096)
def _pieces(x, dps):
    """Breakpoints for both halves at levels of log u (low precision)."""
    extra = int(max(0, mp.log10(abs(x)))) if x != 0 else 0
    with mp.workdps(20 + extra):
        x = mp.mpf(x)
        cut = mp.log(max(mp.exp(-x - 1), 1) + 1.15 * dps * mp.log(10) + 30)
        levels = sorted(set([-mp.mpf(2) ** k for k in range(0, 12)] +
                            [mp.mpf(0), mp.log(3), mp.log(10), mp.log(30), mp.log(100),
                             mp.log(300)] +
                            [mp.log(mp.exp(-x - 1) + k) for k in (0.01, 0.1, 1, 3, 10, 30, 100, 300)] +
                            [cut]))
        half = mp.pi / 2
        vh = _logu_theta(half, x)
        v0 = -x - 1
        th = [mp.mpf(0)]
        ep = [mp.mpf(0)]
        tiny = mp.mpf(10) ** (-(extra + 30))
        for L in levels:
            if v0 < L < vh:
                th.append(_solve(lambda t: _logu_theta(t, x), L, tiny, half, True))
            elif L > vh:
                ep.append(_solve(lambda e: _logu_eps(e, x), L, tiny, half, False))
        th.append(half); ep.append(half)
        return sorted(set(th)), sorted(set(ep)), cut

def _integral(x, kind):
    dps = mp.mp.dps
    th, ep, cut = _pieces(x, dps)
    extra = int(max(0, mp.log10(abs(x)))) if x != 0 else 0
    with mp.workdps(dps + extra + 10):
        X = mp.mpf(x)
        th = [mp.mpf(t) for t in th]; ep = [mp.mpf(e) for e in ep]
        def g(v):
            u = mp.exp(v)
            if kind == "pdf":
                return u * mp.exp(-u)
            if kind == "cdf":
                return mp.exp(-u)
            return -mp.expm1(-u)
        def ft(t):
            return g(_logu_theta(t, X))
        def fe(e):
            if e == 0:
                return mp.mpf(1) if kind == "sf" else mp.mpf(0)
            v = _logu_eps(e, X)
            if v > cut + 1:
                return mp.mpf(1) if kind == "sf" else mp.mpf(0)
            return g(v)
        # theta half: drop pieces where log u > cut (negligible for pdf/cdf, 1 for sf)
        s = mp.mpf(0)
        for a, b in zip(th[:-1], th[1:]):
            if kind != "sf" and _logu_theta(a, X) > cut:
                continue
            s += mp.quad(ft, [a, b])
        for a, b in zip(ep[:-1], ep[1:]):
            if _logu_eps(b, X) > cut:          # whole piece beyond the cut
                if kind == "sf":
                    s += b - a
                continue
            s += mp.quad(fe, [a, b])
        r = s / mp.pi
    return +r

def pdf(x): return _integral(x, "pdf")
def cdf(x): return _integral(x, "cdf")
def sf(x): return _integral(x, "sf")
