"""mpmath reference for scipy.stats.gausshyper (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a, b, c, z) = x**(a-1) (1-x)**(b-1) (1+z x)**(-c) / (B(a, b) H),
    H = 2F1(c, a; a+b; -z),   0 <= x <= 1, a, b > 0, c real, z > -1.

Only the pdf is implemented in SciPy, so only pdf (and logpdf, for the
opportunities section) are provided.  No cdf reference: the cdf is an Appell
F1 function and ``mp.appellf1`` (or ``mp.quad`` of the pdf) at 400 digits
takes minutes per point, and the harness evaluates the cdf at every x for
region labels.  The normalization (pdf integrates to 1) is checked
separately in ``scratch/gausshyper/check_norm.py``.

The normalization constant B(a, b) * H depends only on the shapes and is
memoized (``mp.hyp2f1`` takes up to ~20 s at 400 digits for a = b = 300).
"""
import mpmath as mp

DIST = "gausshyper"
METHODS = ["pdf"]

# inventory example; moderate; z near -1 (c > b, H large); large a = b
# (beta(a, b) ~ 5e-182, x**(a-1) underflows); large b; large c with small z
# (hyp2f1 loses accuracy); huge z with c == a (degenerate hyp2f1 case);
# negative c with large z (H ~ 1e176); tiny a, b; c - a = 1e-12 (nearly
# degenerate hyp2f1); large c with z = 1 (hyp2f1 returns nan); large c with
# z < 0 (hyp2f1 accurate, isolates the (1 + z*x)**c factor).
PARAMS = [(13.7637716041307, 3.118963664868143, 2.514598035018302, 5.1811649903971615),
          (2.0, 3.0, 1.0, 0.5),
          (1.5, 2.5, 3.0, -0.999999),
          (300.0, 300.0, 5.0, 2.0),
          (2.0, 1e4, 3.0, 0.5),
          (3.0, 2.0, 1e4, 1e-4),
          (2.5, 1.5, 2.5, 1e6),
          (2.0, 3.0, -60.0, 1e3),
          (0.01, 0.01, 2.0, 10.0),
          (2.5, 1.5, 2.500000000001, 1e6),
          (3.0, 2.0, 1e4, 1.0),
          (3.0, 2.0, 1e3, -0.5)]

_NORM = {}


def _lognorm(a, b, c, z):
    key = (mp.nstr(a, 30), mp.nstr(b, 30), mp.nstr(c, 30), mp.nstr(z, 30), mp.mp.dps)
    if key not in _NORM:
        logB = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
        if 0 < z <= 10:
            # Pfaff: 2F1(c, a; a+b; -z) = (1+z)**-c 2F1(c, b; a+b; z/(1+z));
            # same value, much faster in mpmath for large a, b
            logH = -c * mp.log1p(z) + mp.log(mp.hyp2f1(c, b, a + b, z / (1 + z)))
        else:
            logH = mp.log(mp.hyp2f1(c, a, a + b, -z))
        _NORM[key] = logB + logH
    return _NORM[key]


def logpdf(x, a, b, c, z):
    x = mp.mpf(x)
    if x <= 0 or x >= 1:
        p = pdf(x, a, b, c, z)
        return mp.log(p) if p != 0 else mp.ninf
    return ((a - 1) * mp.log(x) + (b - 1) * mp.log1p(-x)
            - c * mp.log1p(z * x) - _lognorm(a, b, c, z))


def pdf(x, a, b, c, z):
    x = mp.mpf(x)
    if x < 0 or x > 1:
        return mp.mpf(0)
    if x == 0:
        if a == 1:
            return mp.exp(-_lognorm(a, b, c, z))
        return mp.inf if a < 1 else mp.mpf(0)
    if x == 1:
        if b == 1:
            return mp.exp(-c * mp.log1p(z) - _lognorm(a, b, c, z))
        return mp.inf if b < 1 else mp.mpf(0)
    return mp.exp(logpdf(x, a, b, c, z))


REFS = {"pdf": pdf, "logpdf": logpdf}


def x_points(a, b, c, z):
    # mode region, mean +- a few sd of the beta part (large shapes), and the
    # x where x**(a-1) resp. (1-x)**(b-1) crosses the underflow limit.
    import numpy as np
    a, b = float(a), float(b)
    m = a / (a + b)
    s = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    pts = [m + k * s for k in (-20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20)]
    for p, lim in ((a - 1, 708.0), (b - 1, 708.0)):
        if p > 1:
            e = np.exp(-lim / p)
            pts += [e * 0.5, e * 0.8, e, e * 1.2, e * 2]
            pts += [1 - e * 0.5, 1 - e * 0.8, 1 - e, 1 - e * 1.2, 1 - e * 2]
    pts += [1e-3, 1e-2, 0.05, 0.1, 0.25, 0.75, 0.9, 0.99, 0.999]
    pts = np.array(pts)
    return pts[(pts > 0) & (pts < 1)]
