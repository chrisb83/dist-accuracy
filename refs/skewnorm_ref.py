"""Reference module for scipy.stats.skewnorm.

Docstring pdf: f(x; a) = 2 * phi(x) * Phi(a x), x real, a real.

cdf: for x <= 0 an Owen's T integral form with positive terms only (see _G),
so the deep left tail with a > 0, where Phi(x) - 2 T(x, a) would cancel
catastrophically even at 400 digits, keeps full relative accuracy.
For x > 0, cdf = 1 - G(-x, -a) using the symmetry f(-t; -a) = f(t; a),
and sf(x, a) = cdf(-x, -a).
ppf/isf: Newton on log F / log S (log-concave, so monotone convergence),
started from SciPy's value; much faster than invert() with a quadrature cdf.
DPS = 100 because the cdf is a quadrature (harness checks at 200 still pass).
Quantile runs are slow (~7 s per condition number): split --params groups.
"""
import mpmath as mp


DIST = "skewnorm"
DPS = 100

# example; small and moderate a; large a (near half-normal, Owen's T with
# large a); tiny a; negative a (mirror image).
PARAMS = [(4.0,), (1.0,), (10.0,), (100.0,), (1e4,), (1e-8,), (-4.0,)]


def _logncdf(u):
    """log Phi(u); asymptotic series for u < -1e10 (mp.erfc overflows for
    |u| ~ 1e154 and beyond)."""
    u = mp.mpf(u)
    if u > -1e10:
        if u > 1e10:
            return mp.mpf(0)          # -phi(u)/u < exp(-5e19)
        return mp.log(mp.ncdf(u))
    z = 1 / (u * u)
    term, tot, n = mp.mpf(1), mp.mpf(1), 1
    while abs(term) > mp.mpf(10) ** (-mp.mp.dps - 5):
        term *= -(2 * n - 1) * z
        tot += term
        n += 1
    return -u * u / 2 - mp.log(-u) - mp.log(2 * mp.pi) / 2 + mp.log(tot)


def logpdf(x, a):
    x = mp.mpf(x)
    return mp.log(2) - x * x / 2 - mp.log(2 * mp.pi) / 2 + _logncdf(a * x)


def pdf(x, a):
    return mp.exp(logpdf(x, a))


def _geom(lo, hi, start, w, down=False):
    """Breakpoints start +- w*4**k strictly inside (lo, hi)."""
    out = []
    for k in range(12):
        p = start - w * 4 ** k if down else start + w * 4 ** k
        if lo < p < hi:
            out.append(p)
    return out


def _quad_pts(f, lo, hi, extra):
    pts = sorted(set([lo, hi] + extra))
    return mp.quad(f, pts)


def _G(x, a):
    """cdf for x <= 0 without cancellation.

    F(x; a) = Phi(x) - 2 T(x, a), T(h, a) = 1/(2 pi) int_0^a g(t) dt,
    g(t) = exp(-h^2 (1 + t^2)/2) / (1 + t^2).  With T(h, inf) = Phi(-|h|)/2:
      a >= 0:  F = 1/pi int_a^inf g(t) dt                 (positive integrand)
      a <  0:  F = Phi(x) + 1/pi int_0^{|a|} g(t) dt      (sum of positives)
    The t-integral is cut where g drops below 10**-(dps+25) of its maximum and
    evaluated as int d theta (t = tan theta, t <= 1) plus int d phi
    (t = cot phi, t > 1); both integrands are bounded and smooth.  mp.quad's
    tolerance is absolute (10**-dps), so integrands are normalized to 1 at
    their maximum and the factor is applied afterwards.
    Near x = 0 (|x| (1 + |a|) < 1e-2): F = F(0) + int_0^x pdf, F(0) = 1/2 - atan(a)/pi.
    """
    x = mp.mpf(x)
    a = mp.mpf(a)
    if abs(x) * (1 + abs(a)) < mp.mpf("1e-2"):
        F0 = mp.mpf(1) / 2 - mp.atan(a) / mp.pi
        if x == 0:
            return F0
        return F0 + mp.quad(lambda t: pdf(t, a), [0, x])
    h2 = x * x
    e0 = -h2 / 2
    L = mp.mpf(2.4) * mp.mp.dps + 60                 # ln of the cut ratio
    one = mp.mpf(1)
    q4 = mp.pi / 4
    if a >= 0:
        A = 1 + a * a
        T = mp.sqrt(a * a + L / -e0)                 # cut in t
        w = 1 / (1 + h2 * A * a + abs(x))            # decay scale in theta/phi
        tot = mp.mpf(0)
        if a < 1:                                    # theta part, t in [a, min(T, 1)]
            lo, hi = mp.atan(a), mp.atan(min(T, one))
            f = lambda th: mp.exp(e0 * (mp.tan(th) ** 2 - a * a))
            tot += _quad_pts(f, lo, hi, _geom(lo, hi, lo, w))
        if T > 1:                                    # phi part, t in [max(a, 1), T]
            lo, hi = mp.atan(1 / T), mp.atan(1 / max(a, one))
            f = lambda ph: mp.exp(e0 * (1 / mp.tan(ph) ** 2 - a * a))
            ex = _geom(lo, hi, hi, w, down=True) if a >= 1 else []
            tot += _quad_pts(f, lo, hi, ex)
        return mp.exp(e0 * A) * tot / mp.pi
    b = min(-a, mp.sqrt(L / -e0))
    w = 1 / (1 + abs(x))
    tot = mp.mpf(0)
    lo, hi = mp.mpf(0), mp.atan(min(b, one))
    f = lambda th: mp.exp(e0 * mp.tan(th) ** 2)
    tot += _quad_pts(f, lo, hi, _geom(lo, hi, lo, w))
    if b > 1:
        lo, hi = mp.atan(1 / b), q4
        f = lambda ph: mp.exp(e0 / mp.tan(ph) ** 2)
        tot += _quad_pts(f, lo, hi, _geom(lo, hi, hi, w, down=True))
    return mp.exp(_logncdf(x)) + mp.exp(e0) * tot / mp.pi


def cdf(x, a):
    x = mp.mpf(x)
    if x <= 0:
        return _G(x, a)
    return 1 - _G(-x, -a)


def sf(x, a):
    return cdf(-mp.mpf(x), -mp.mpf(a))


def _newton_quantile(q, a, upper, x0):
    """Solve log F(x) = log q (upper=False) or log S(x) = log q (upper=True).

    log F and log S are concave (the skew-normal density is log-concave), so
    Newton's method converges monotonically after the first step from any
    start; this needs ~10 cdf evaluations instead of ~50 for invert().
    """
    q = mp.mpf(q)
    if mp.mp.dps > 110:                  # converge cheaply first, then refine
        with mp.workdps(100):
            x0 = _newton_quantile(q, a, upper, x0)
    lq = mp.log(q)
    x = mp.mpf(x0)
    tol = mp.mpf(10) ** (-min(mp.mp.dps // 2 + 10, 60))
    for _ in range(200):
        if upper:
            P = sf(x, a)
            step = (mp.log(P) - lq) * P / pdf(x, a)      # x += step
        else:
            P = cdf(x, a)
            step = -(mp.log(P) - lq) * P / pdf(x, a)
        x += step
        if abs(step) <= tol * (1 + abs(x)):
            break
    else:
        raise RuntimeError("no convergence")
    return x


def _start(q, a, upper):
    import numpy as np
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(stats.skewnorm.isf(float(q), float(a)) if upper
                  else stats.skewnorm.ppf(float(q), float(a)))
    return v if np.isfinite(v) else 0.0


def ppf(q, a):
    q = mp.mpf(q)
    if q <= 0.5:
        return _newton_quantile(q, a, False, _start(q, a, False))
    return _newton_quantile(1 - q, a, True, _start(q, a, False))


def isf(q, a):
    q = mp.mpf(q)
    if q <= 0.5:
        return _newton_quantile(q, a, True, _start(q, a, True))
    return _newton_quantile(1 - q, a, False, _start(q, a, True))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    import numpy as np
    pts = [-40.0, -38.0, -30.0, -20.0, -8.0, -5.0, -3.0, -1.5, -0.5, -0.1,
           0.1, 0.5, 1.5, 3.0, 5.0, 8.0, 20.0, 30.0, 38.0, 40.0,
           1.2e154, 1.5e154, -1.5e154]
    if a != 0:
        # where a*x reaches the ndtr tail / log_ndtr regimes and where the
        # cdf is near the 1e-6 switch for a > 0 (x < 0) and a < 0 (x > 0)
        for t in [-40.0, -30.0, -20.0, -10.0, -5.0, -3.0, -2.0, -1.0, -0.3]:
            pts.append(t / abs(a))
            pts.append(-t / abs(a))
    return np.array(pts)
