"""Independent nct reference: T = (Z + d)/S, S = sqrt(V/k).
cdf(t) = E[Phi(t S - d)], sf(t) = E[Phi(d - t S)], pdf(t) = E[S phi(t S - d)].
Integrated with mp.quad in u = log S, with peak/bound location (no Poisson series, no 1F1)."""
import mpmath as mp

def _Lchi(u, k):
    return (k/2)*mp.log(k) - (k/2 - 1)*mp.log(2) - mp.loggamma(k/2) + k*u - k*mp.exp(2*u)/2

def _logncdf(z):
    if z < -1e10:  # asymptotic Mills series, terms < z^-60 ~ 1e-600
        z2 = z*z; term = mp.mpf(1); ssum = mp.mpf(1)
        for n in range(1, 31):
            term *= -(2*n - 1)/z2; ssum += term
        return -z2/2 - mp.log(-z) - mp.log(2*mp.pi)/2 + mp.log(ssum)
    if z > 1e10:
        return mp.mpf(0) if z > 1e100 else mp.log1p(-mp.exp(_logncdf(-z)))
    if z < -5:
        return mp.log(mp.erfc(-z/mp.sqrt(2))/2)
    return mp.log(mp.ncdf(z))

def _L(kind, u, t, k, d):
    s = mp.exp(u)
    if kind == 'cdf':
        return _Lchi(u, k) + _logncdf(t*s - d)
    if kind == 'sf':
        return _Lchi(u, k) + _logncdf(d - t*s)
    z = t*s - d
    return _Lchi(u, k) + u - z*z/2 - mp.log(2*mp.pi)/2

def value(kind, t, k, d, drop=250):
    t, k, d = mp.mpf(t), mp.mpf(k), mp.mpf(d)
    L = lambda u: _L(kind, u, t, k, d)
    # coarse grid at the working precision
    with mp.workdps(30):
        us = [mp.mpf(-1600) + j*mp.mpf('0.5') for j in range(3281)]
        vals = [L(u) for u in us]
    us = [mp.mpf(u) for u in us]
    j = max(range(len(vals)), key=lambda i: vals[i])
    a, b = us[max(j-1, 0)], us[min(j+1, len(us)-1)]
    # golden section for the max in [a, b] (zoom on narrow peaks)
    g = (mp.sqrt(5) - 1)/2
    for _ in range(400):
        c1, c2 = b - g*(b - a), a + g*(b - a)
        if L(c1) > L(c2): b = c2
        else: a = c1
        if b - a < mp.mpf(10)**(-mp.mp.dps//2): break
    up = (a + b)/2; Lm = L(up)
    def bound(sign):
        h = mp.mpf(10)**(-mp.mp.dps//3)
        while L(up + sign*h) > Lm - drop:
            h *= 2
            if h > 3000: return up + sign*h
        lo, hi = h/2, h
        for _ in range(200):
            m = (lo + hi)/2
            if L(up + sign*m) > Lm - drop: lo = m
            else: hi = m
        return up + sign*hi
    ulo, uhi = bound(-1), bound(+1)
    pts = [ulo + (up - ulo)*i/12 for i in range(12)] + [up + (uhi - up)*i/12 for i in range(13)]
    I = mp.quad(lambda u: mp.exp(L(u) - Lm), pts)
    return I*mp.exp(Lm)

def both(kind, t, k, d):
    r = []
    for dps in (100, 200):
        with mp.workdps(dps):
            r.append(value(kind, t, k, d))
    with mp.workdps(200):
        rel = abs(r[0] - r[1])/abs(r[1])
    return r[1], rel
