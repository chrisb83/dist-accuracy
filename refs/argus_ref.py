"""Reference module for scipy.stats.argus.

Standard form, shape chi > 0, support 0 < x < 1. Docstring pdf:
    f(x, chi) = chi**3 / (sqrt(2 pi) Psi(chi)) x sqrt(1 - x**2) exp(-chi**2 (1 - x**2) / 2)
    Psi(chi) = Phi(chi) - chi phi(chi) - 1/2 = P(3/2, chi**2/2) / 2
with P the regularized lower incomplete gamma function. With a = chi**2/2 and
b = a (1 - x**2):
    sf(x)  = P(3/2, b) / P(3/2, a)
    cdf(x) = (P(a) - P(b)) / P(a) = (Q(b) - Q(a)) / P(a)
The cdf is computed from P(a) - P(b) (small a) or Q(b) - Q(a) (a >= 1, where
P(a) - P(b) would need ~a/2.3 extra digits), with extra working digits for the
cancellation when x is small (relative difference ~ x**2). Q(3/2, t) uses the
closed form erfc(sqrt t) + 2 sqrt(t/pi) exp(-t) (both terms positive).
ppf/isf are not implemented by argus (generic), so they are not referenced.
"""
import mpmath as mp

DIST = "argus"

# Example value; small chi (log cancellation in logpdf A term), chi where
# _argus_phi ~ chi**3 underflows (chi < ~1e-103) and where chi**2 underflows
# (chi < 1e-162), chi = 1e-100 (sf underflows near x = 1); moderate and large chi (cdf tiny away from x = 1);
# chi**2 overflow (chi > 1.34e154).
PARAMS = [(1.0,), (1e-3,), (1e-50,), (1e-100,), (1e-110,), (1e-200,), (0.1,), (10.0,),
          (100.0,), (1e4,), (1e155,)]


def _P(t):
    return mp.gammainc(mp.mpf(1.5), 0, t, regularized=True)


def _Q(t):
    if t > 1e6:  # mpmath erfc cannot handle huge arguments; asymptotic path is fast
        return mp.gammainc(mp.mpf(1.5), t, mp.inf, regularized=True)
    return mp.erfc(mp.sqrt(t)) + 2 * mp.sqrt(t / mp.pi) * mp.exp(-t)


def pdf(x, chi):
    a = chi ** 2 / 2
    psi = _P(a) / 2
    return (chi ** 3 / (mp.sqrt(2 * mp.pi) * psi) * x * mp.sqrt(1 - x ** 2)
            * mp.exp(-a * (1 - x ** 2)))


def logpdf(x, chi):
    a = chi ** 2 / 2
    psi = _P(a) / 2
    return (3 * mp.log(chi) - mp.log(mp.sqrt(2 * mp.pi) * psi) + mp.log(x)
            + mp.log(1 - x ** 2) / 2 - a * (1 - x ** 2))


def sf(x, chi):
    a = chi ** 2 / 2
    return _P(a * (1 - x ** 2)) / _P(a)


def cdf(x, chi):
    extra = 20
    if 0 < x < 1:
        extra += int(-2 * mp.log10(x)) + 5
    with mp.workdps(mp.mp.dps + extra):
        x = mp.mpf(x)
        chi = mp.mpf(chi)
        a = chi ** 2 / 2
        b = a * (1 - x ** 2)
        if a < 1:
            num = _P(a) - _P(b)
        else:
            num = _Q(b) - _Q(a)
        r = num / _P(a)
    return +r


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, chi: mp.log(cdf(x, chi)),
    "sf": sf,
    "logsf": lambda x, chi: mp.log(sf(x, chi)),
}


def x_points(chi):
    # Ends of the support (x -> 0: cdf = 1 - sf cancels; x -> 1: 1 - x*x),
    # and the scale of the pdf: for large chi the mass sits at 1 - x ~ 1/chi**2.
    import numpy as np
    pts = [0.5, 0.1, 0.9, 1e-3, 1e-5, 1e-8, 1e-10, 1e-12, 1e-16, 1e-30, 1e-100]
    for d in [1e-16, 3e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts.append(1 - d)
    chi = float(chi)
    if 1e-100 < chi < 1e100:
        for k in [0.01, 0.1, 1.0, 10.0, 100.0]:
            v = 1 - k / chi ** 2
            if 0 < v < 1:
                pts.append(v)
            v = k / chi
            if 0 < v < 1:
                pts.append(v)
    return np.array(pts)
