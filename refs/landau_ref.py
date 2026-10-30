"""Reference module for scipy.stats.landau.

Parametrization
---------------
SciPy delegates to Boost.Math ``landau_distribution`` (location 0, scale 1),
whose standard density is

    f(x) = 1/pi * int_0^inf exp(-t) cos(x t + (2/pi) t log t) dt,

with sf(x) ~ 2/(pi x) for x -> inf.  The docstring formula
f_L(y) = 1/pi int_0^inf exp(-t log t - y t) sin(pi t) dt is Landau's original
form, which is a *different* standardization (sf_L(y) ~ 1/y).  Comparing
characteristic functions gives

    X_scipy = (2/pi) (X_L + log(2/pi)),   i.e.
    f(x) = (pi/2) f_L(y),  F(x) = F_L(y),  y = pi x / 2 + log(pi/2).

(Checked: agreement with SciPy to ~1e-16 in the bulk, see the self-checks.)

Non-oscillatory integral for f_L, F_L
-------------------------------------
f_L is the inverse Laplace transform of exp(s log s).  Deforming the Bromwich
line onto the steepest-descent path of phi(s) = s log s + y s,
s(theta) = r(theta) exp(i theta), r = exp(-y - theta cot theta), on which
phi = -u(theta) is real, gives (Zolotarev's form for the alpha = 1 stable law)

    u(theta) = theta / sin(theta) * exp(-y - theta cot theta),  0 < theta < pi,
    f_L(y) = 1/pi int_0^pi u exp(-u) dtheta,
    F_L(y) = 1/pi int_0^pi exp(-u) dtheta,
    sf_L(y) = 1/pi int_0^pi -expm1(-u) dtheta.

All integrands are positive and smooth; u increases monotonically from
exp(-y-1) (theta = 0) to inf (theta = pi).  The interval is split at pi/2;
the upper half is parametrized by eps = pi - theta so that the right tail
(theta -> pi, eps ~ 2/y) keeps full relative accuracy.  Breakpoints are placed
where log u crosses fixed levels (found by bisection at low precision), and
pieces where u exceeds a cut (exp(-u) below 10**-(1.15 dps) relative) are
dropped (pdf, cdf) or integrate to their length (sf).

Precision: -y - theta cot theta cancels to O(1) for large y, losing log10|y|
digits; the integrals are therefore evaluated with log10|y| + 10 extra digits.

ppf / isf: Newton iteration on log F (q <= 1/2) or log sf (q > 1/2), started
from SciPy's own value (only a starting point; convergence to the true root is
checked) with the bracketing ``accuracy.invert`` as fallback.
"""
import functools
import math

import mpmath as mp

DIST = "landau"
PARAMS = [()]
DPS = 100


def _y(x):
    return mp.pi * x / 2 + mp.log(mp.pi / 2)


def _vt(t, y):          # log u, theta in [0, pi/2]
    if t == 0:
        return -y - 1
    return mp.log(t / mp.sin(t)) - y - t * mp.cot(t)


def _ve(e, y):          # log u, theta = pi - eps, eps in (0, pi/2]
    return mp.log((mp.pi - e) / mp.sin(e)) - y + (mp.pi - e) * mp.cot(e)


def _extra(x):
    ax = abs(float(x))
    return (int(math.log10(ax)) if ax > 1 else 0) + 10


def _solve(fn, L, a, b, inc):
    """Point where the monotone fn crosses level L: |fn - L| < tol, i.e.
    u within 2% of exp(L) for L <= 0 and within 0.02 absolute for L > 0."""
    tol = 0.02 * min(1, mp.exp(-L))
    m = (a + b) / 2
    for _ in range(600):                          # geometric bisection
        if b < 4 * a:
            break
        m = mp.sqrt(a * b)
        v = fn(m)
        if abs(v - L) < tol:
            return m
        if (v < L) == inc:
            a = m
        else:
            b = m
    fa, fb = fn(a) - L, fn(b) - L                 # Illinois regula falsi
    side = 0
    for _ in range(200):
        m = (a * fb - b * fa) / (fb - fa)
        fm = fn(m) - L
        if abs(fm) < tol:
            return m
        if (fm > 0) == (fb > 0):
            b, fb = m, fm
            if side == 1:
                fa /= 2
            side = 1
        else:
            a, fa = m, fm
            if side == -1:
                fb /= 2
            side = -1
    return m


@functools.lru_cache(maxsize=None)
def _pieces(x, dps):
    with mp.workdps(20 + _extra(x)):
        y = _y(x)
        v0 = -y - 1
        cutu = max(mp.exp(v0), 1) + 1.15 * dps * mp.log(10) + 30
        cut = mp.log(cutu)
        # lowest level: below it u < exp(lmin), so the pdf and sf integrands
        # (<= u) contribute < pi*exp(lmin) << 10**-(dps+10) * result, using
        # f_L(y) >= 0.1/y^2 and sf_L(y) >= 0.1/y for y >= 1.
        lmin = -(dps + 10) * mp.log(10) - 2 * mp.log(max(abs(y), 1)) - 10
        lv = [-mp.mpf(2) ** k for k in range(0, 40) if 2 ** k < -lmin]
        lv += [lmin, mp.mpf(0), mp.mpf(1), mp.mpf(2)]
        if v0 > 0:
            s0 = mp.exp(v0)
            lv += [mp.log(s0 + k) for k in (0.3, 3, 30)]
        lv = sorted(set(L for L in lv if L < cut)) + [cut]
        half = mp.pi / 2
        vh = _vt(half, y)
        tiny = mp.mpf(10) ** (-(_extra(x) + 30))
        th, ep = [mp.mpf(0), half], [mp.mpf(0), half]
        for L in lv:
            if v0 < L < vh:
                th.append(_solve(lambda t: _vt(t, y), L, tiny, half, True))
            elif L > vh:
                ep.append(_solve(lambda e: _ve(e, y), L, tiny, half, False))
        return tuple(sorted(set(th))), tuple(sorted(set(ep))), cut, lmin


@functools.lru_cache(maxsize=None)
def _integral(x, kind, dps):
    """1/pi * integral for f_L ('pdf'), F_L ('cdf'), sf_L ('sf') at y(x)."""
    th, ep, cut, lmin = _pieces(x, dps)
    with mp.workdps(dps + _extra(x)):
        y = _y(x)
        one = mp.mpf(1) if kind == "sf" else mp.mpf(0)

        def g(v):
            if v > cut + 3:
                return one
            u = mp.exp(v)
            if kind == "pdf":
                return u * mp.exp(-u)
            if kind == "cdf":
                return mp.exp(-u)
            return -mp.expm1(-u)

        def piece(h, a, b):
            # mp.quad stops on an *absolute* error estimate (~10**-dps), so a
            # piece worth 1e-264 would be accepted at the lowest degree.  Map
            # to [0, 1] and divide by the largest of three samples so every
            # piece integral is O(1)..O(1e-3) before calling quad.
            # The quadrature itself runs at the requested dps (nodes z need no
            # more); only the integrand uses the extra digits, which are needed
            # for -y - theta cot theta, not for the nodes.
            w = b - a
            M = max(h(a), h(a + w / 2), h(b))
            if M == 0:
                return mp.mpf(0)
            hp = mp.mp.dps

            def hz(z):
                with mp.workdps(hp):
                    return h(a + w * z) / M
            with mp.workdps(dps):
                r = mp.quad(hz, [0, 1])
            return w * M * r

        s = mp.mpf(0)
        low = lmin + mp.mpf("0.5")
        for a, b in zip(th[:-1], th[1:]):
            if kind != "sf" and _vt(a, y) > cut:
                continue
            if kind != "cdf" and _vt(b, y) < low:   # u < exp(lmin) throughout
                continue
            s += piece(lambda t: g(_vt(t, y)), a, b)
        for a, b in zip(ep[:-1], ep[1:]):
            if _ve(b, y) >= cut - 0.03:           # whole piece beyond the cut
                if kind == "sf":
                    s += b - a
                continue
            if kind != "cdf" and _ve(a, y) < low:
                continue
            s += piece(lambda e: g(_ve(e, y)) if e > 0 else one, a, b)
        r = s / mp.pi
    return r


# median of the SciPy-standard Landau law (switch between cdf and sf integrals)
XMED = 1.3558


def _key(x):
    # |x| below 10**-(dps+20) gives the same y = pi x/2 + log(pi/2) as x = 0 to
    # working precision (difference < 10**-(dps+20) relative): reuse x = 0.
    x = mp.mpf(x)
    if abs(x) < mp.mpf(10) ** (-(mp.mp.dps + 20)):
        x = mp.mpf(0)
    return x, mp.mp.dps


def pdf(x):
    x, d = _key(x)
    if x < -12:                                   # pdf < exp(-1e7): never a double
        return mp.mpf(0)
    return +(mp.pi / 2 * _integral(x, "pdf", d))


def cdf(x):
    x, d = _key(x)
    if x < -12:
        return mp.mpf(0)
    if x <= XMED:
        return +_integral(x, "cdf", d)
    return +(1 - _integral(x, "sf", d))


def sf(x):
    x, d = _key(x)
    if x < -12:
        return mp.mpf(1)
    if x <= XMED:
        return +(1 - _integral(x, "cdf", d))
    return +_integral(x, "sf", d)


def _newton(q, upper):
    """Solve F(x) = q (upper=False, log F) or sf(x) = q (upper=True, log sf)."""
    from scipy import stats
    q = mp.mpf(q)
    qf = float(q)
    x0 = stats.landau.isf(qf) if upper else stats.landau.ppf(qf)
    if not math.isfinite(x0):
        return None
    x = mp.mpf(x0)
    lq = mp.log(q)
    # quadratic convergence: after a step of relative size 10**-(0.3 dps) the
    # remaining error is ~10**-(0.6 dps)
    tol = mp.mpf(10) ** (-int(0.3 * mp.mp.dps))
    f = None
    for _ in range(12):
        P = sf(x) if upper else cdf(x)
        if f is None:                             # chord method: once the
            f = pdf(x)                            # step is < 1e-8 the old pdf
        if P <= 0 or f <= 0:                      # still converges fast
            return None
        step = (mp.log(P) - lq) * P / f
        x = x + step if upper else x - step
        if abs(step) <= tol * max(1, abs(x)):
            return x
        if abs(step) > mp.mpf("1e-8") * max(1, abs(x)):
            f = None
    return None


def _inv(q, upper):
    x = _newton(q, upper)
    if x is not None:
        return x
    from accuracy import invert
    if upper:
        return invert(sf, q, increasing=False, positive=False)
    return invert(cdf, q, positive=False)


def ppf(q):
    q = mp.mpf(q)
    if q > 0.5:
        return _inv(1 - q, True)                  # 1 - q exact in mpmath
    return _inv(q, False)


def isf(q):
    q = mp.mpf(q)
    if q > 0.5:
        return _inv(1 - q, False)
    return _inv(q, True)


REFS = {
    "pdf": pdf,
    "cdf": cdf,
    "sf": sf,
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # Left tail (double-exponential decay): pdf/cdf underflow near x ~ -5.1;
    # mode near -0.43; right tail power law (pdf ~ 2/(pi x^2) subnormal for
    # x > ~1e154).
    import numpy as np
    left = [-0.5, -1.5, -2.5, -3.0, -3.5, -4.0, -4.25, -4.5, -4.75, -5.0, -5.05, -5.1]
    right = [1.3558, 3.0, 20.0, 50.0, 200.0, 1e3, 1e5, 1e7, 1e15, 1e154, 1e155]
    return np.array(left + right)


def q_points():
    import numpy as np
    return np.array([1e-308, 1e-250, 1e-150, 1e-50, 1e-30, 1e-6, 0.25, 0.75,
                     1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2.2e-16])
