import mpmath as mp, functools

CNT = [0]
def _y(x):
    # SciPy/Boost standard form -> Landau's original variable
    return mp.pi * x / 2 + mp.log(mp.pi / 2)

def _vt(t, y):   # log u, theta in [0, pi/2]
    if t == 0:
        return -y - 1
    return mp.log(t / mp.sin(t)) - y - t * mp.cot(t)

def _ve(e, y):   # log u, theta = pi - eps
    return mp.log((mp.pi - e) / mp.sin(e)) - y + (mp.pi - e) * mp.cot(e)

def _solve(fn, L, a, b, inc):
    fa = None
    for _ in range(400):
        m = (a + b) / 2 if b < 4 * a else mp.sqrt(a * b)
        v = fn(m)
        if abs(v - L) < mp.mpf('0.02'):
            return m
        if (v < L) == inc:
            a = m
        else:
            b = m
    return m

def _extra(y):
    return int(max(0, float(mp.log10(abs(y))))) + 10 if y != 0 else 10

@functools.lru_cache(maxsize=8192)
def _pieces(ys, dps):
    y = mp.mpf(ys)
    with mp.workdps(20 + _extra(y)):
        y = mp.mpf(ys)
        v0 = -y - 1
        cutu = max(mp.exp(v0), 1) + 1.15 * dps * mp.log(10) + 30
        cut = mp.log(cutu)
        lv = [-mp.mpf(4) ** k for k in range(0, 6)] + [mp.mpf(0), mp.mpf(1), mp.mpf(2)]
        if v0 > 0:
            s0 = mp.exp(v0)
            lv += [mp.log(s0 + k) for k in (1, 10, 100)]
        lv = sorted(set(L for L in lv if L < cut)) + [cut]
        half = mp.pi / 2
        vh = _vt(half, y)
        tiny = mp.mpf(10) ** (-(_extra(y) + 30))
        th, ep = [mp.mpf(0), half], [mp.mpf(0), half]
        for L in lv:
            if v0 < L < vh:
                th.append(_solve(lambda t: _vt(t, y), L, tiny, half, True))
            elif L > vh:
                ep.append(_solve(lambda e: _ve(e, y), L, tiny, half, False))
        return tuple(sorted(set(th))), tuple(sorted(set(ep))), cut

def _integral(x, kind, method='tanh-sinh'):
    dps = mp.mp.dps
    y = _y(x)
    th, ep, cut = _pieces(mp.nstr(y, 60 + _extra(y)) if False else y, dps)
    with mp.workdps(dps + _extra(y)):
        y = _y(mp.mpf(x))
        def g(v):
            CNT[0] += 1
            if v > cut + 3:
                return mp.mpf(1) if kind == "sf" else mp.mpf(0)
            u = mp.exp(v)
            if kind == "pdf":
                return u * mp.exp(-u)
            if kind == "cdf":
                return mp.exp(-u)
            return -mp.expm1(-u)
        s = mp.mpf(0)
        for a, b in zip(th[:-1], th[1:]):
            if kind != "sf" and _vt(a, y) > cut:
                continue
            s += mp.quad(lambda t: g(_vt(t, y)), [a, b], method=method)
        for a, b in zip(ep[:-1], ep[1:]):
            if a == 0 or _ve(a, y) >= cut - mp.mpf('1e-6'):
                if _ve(b, y) >= cut - 0.03:
                    if kind == "sf":
                        s += b - a
                    continue
            s += mp.quad(lambda e: g(_ve(e, y)) if e > 0 else (1 if kind == 'sf' else 0), [a, b], method=method)
        r = s / mp.pi
        if kind == "pdf":
            r = r * mp.pi / 2
    return +r
