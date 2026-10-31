# Independent verifier reference: docstring pdf f = k/((x^2-rho^2)^2 + rho^2),
# rewritten in u = x - rho (x^2 - rho^2 = u*(u + 2 rho)) so that the peak is
# resolvable at any dps; cdf by mp.quad of the pdf in u (complement
# 1 - int_u^inf beyond the peak). Different route from the analyst.
import mpmath as mp

def _k(rho):
    s = mp.sqrt(rho**2 + 1)
    return 2*mp.sqrt(2)*rho**2*s/(mp.pi*mp.sqrt(rho**2 + rho*s))

def _diff(x, rho):
    with mp.workdps(2000):      # exact difference of two doubles
        d = mp.mpf(x) - mp.mpf(rho)
    return +d

def _g(u, rho, k):
    return k/((u*(u + 2*rho))**2 + rho**2)

def pdf(x, rho):
    rho = mp.mpf(rho)
    return _g(_diff(x, rho), rho, _k(rho))

def _pts(rho):
    P = set()
    for j in range(-330, 331, 3):
        P.add(mp.mpf(10)**j); P.add(-mp.mpf(10)**j)
    return P

def squad(f, pts):
    """mp.quad per segment with the integrand scaled to O(1) on each segment
    (mp.quad's tolerance is absolute, so tiny integrands are otherwise
    accepted far too early)."""
    tot = mp.mpf(0)
    for a, b in zip(pts[:-1], pts[1:]):
        if a == b:
            continue
        cand = [f(a)] if a != -mp.inf else []
        if b != mp.inf:
            cand.append(f(b))
            if a != -mp.inf:
                cand.append(f((a + b)/2))
        sc = max(cand)
        if b == mp.inf:
            # t = a + |a| s (a > 0 here), s in [0, inf)
            w = abs(a)
            tot += sc*w*mp.quad(lambda s: f(a + w*s)/sc, [0, 1, 10, mp.inf])
        else:
            # map to [0, 1]: mp.quad's tolerance is absolute in the result
            w = b - a
            tot += sc*w*mp.quad(lambda s: f(a + w*s)/sc, [0, 1])
    return tot

def _int(rho, a, b):
    k = _k(rho)
    pts = [a] + sorted(p for p in _pts(rho) if a < p < b) + [b]
    return squad(lambda u: _g(u, rho, k), pts)

def cdf(x, rho):
    rho = mp.mpf(rho)
    d = _diff(x, rho)
    if d <= 0:
        # far left of the peak: integrate in x itself (u-nodes near -rho
        # would need ~log10(rho/|x|) extra digits)
        k = _k(rho)
        fx = lambda t: k/(((t - rho)*(t + rho))**2 + rho**2)
        x = mp.mpf(x)
        if x <= rho/2:
            xp = [mp.mpf(0)] + sorted(p for p in _pts(rho) if 0 < p < x) + [x]
            return squad(fx, xp)
        xp = [mp.mpf(0)] + sorted(p for p in _pts(rho) if 0 < p < rho/2) + [rho/2]
        return squad(fx, xp) + _int(rho, -rho/2, d)
    return 1 - _int(rho, d, mp.inf)

def total(rho):
    rho = mp.mpf(rho)
    return _int(rho, -rho, mp.inf)
