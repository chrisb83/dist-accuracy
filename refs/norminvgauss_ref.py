"""Reference module for scipy.stats.norminvgauss.

Docstring pdf (standard form), a > 0, |b| < a, x real:
    f(x; a, b) = a K_1(a sqrt(1 + x^2)) / (pi sqrt(1 + x^2)) * exp(sqrt(a^2 - b^2) + b x)
pdf: K_1 by its power series (with extra working precision against the
cancellation for larger z) or by its asymptotic expansion for
z > ~(dps + 10) ln(10) / 2; mp.besselk(1, z) is 1-3 s per call at 100 digits.
cdf/sf: normal mean-variance mixture X | V ~ N(b V, V), V ~ IG(mean 1/gamma,
shape 1), gamma = sqrt(a^2 - b^2) (the representation in the docstring):
    F(x) = int_0^inf Phi((x - b v)/sqrt(v)) f_V(v) dv,
    S(x) = int_0^inf Phi((b v - x)/sqrt(v)) f_V(v) dv,
    f_V(v) = (2 pi v^3)^(-1/2) exp(gamma - 1/(2 v) - gamma^2 v / 2).
In s = log v the integrand is analytic and decays double-exponentially, so
the plain trapezoidal rule converges geometrically; it is applied with step
w/20 (w: peak width from the curvature of the log integrand) until the
integrand is below 1e-70 of its peak, at a fixed internal precision of 60
digits (independent of mp.dps); the step is halved until two successive
results agree to 1e-50 relative (w/20 alone was off by 1.6e-8 for
a = 10, b = 9.99999, see scratch/norminvgauss/recheck). Only the smaller tail is integrated (cdf for
x < mean, sf otherwise); the other is 1 - tail.
isf: safeguarded Newton in u = asinh(x) on log(sf) (q <= 1/2) or log(cdf).
"""
import functools

import mpmath as mp

DIST = "norminvgauss"
DPS = 100

# example; symmetric; strongly skewed (b close to a: exponent b x - a sq
# cancels for large x); negative b; large a (sharp peak, -a*sq + gamma cancels);
# tiny a (Cauchy-like up to |x| ~ 1/a); b/a = 1 - 1e-6; large a and b.
PARAMS = [(1.25, 0.5), (1.0, 0.0), (1.0, 0.999), (2.0, -1.5), (1e4, 0.0),
          (1e-3, 0.0), (10.0, 9.99999), (1e6, 5e5)]

QDPS = 60

def logk1e(z):
    # log(K_1(z) e^z)
    z = mp.mpf(z)
    dps = mp.mp.dps
    if z > (dps + 10) * mp.log(10) / 2 + 5:
        # asymptotic expansion, truncated at the smallest term
        tol = mp.mpf(10) ** (-dps - 5)
        s = term = mp.mpf(1)
        k = 0
        while True:
            k += 1
            new = term * (4 - (2 * k - 1) ** 2) / (k * 8 * z)
            if abs(new) >= abs(term) or abs(new) < tol:
                s += new
                break
            term = new
            s += term
        return mp.log(mp.pi / (2 * z)) / 2 + mp.log(s)
    extra = int(2 * z / mp.log(10)) + 15
    with mp.workdps(dps + extra):
        z = +z
        t = z * z / 4
        # I1 and the psi series together
        term = mp.mpf(1)  # (t^k)/(k!(k+1)!)
        psi1 = -mp.euler          # psi(k+1)
        psi2 = 1 - mp.euler       # psi(k+2)
        sI = mp.mpf(0)
        sP = mp.mpf(0)
        k = 0
        tol = mp.mpf(10) ** (-(dps + extra))
        while True:
            sI += term
            sP += (psi1 + psi2) * term
            k += 1
            term = term * t / (k * (k + 1))
            psi1 += mp.mpf(1) / k
            psi2 += mp.mpf(1) / (k + 1)
            if term < tol * abs(sI) and k > 5:
                break
        I1 = z / 2 * sI
        r = 1 / z + mp.log(z / 2) * I1 - z / 4 * sP
        r = mp.log(r) + z
    return +r


def logpdf(x, a, b):
    # log K_1 = logk1e - z; the large terms -a*sq + gamma + b*x are combined
    # exactly as b*x - a*(sq - 1) - (a - gamma), with sq - 1 = x^2/(sq + 1)
    # and a - gamma = b^2/(a + gamma); extra digits cover the remaining
    # cancellation b*x - a*x^2/(sq + 1) for large a*|x|.
    extra = int(mp.log10(1 + abs(a) * (1 + abs(x)))) + 10
    with mp.workdps(mp.mp.dps + extra):
        x, a, b = +x, +a, +b
        sq = mp.sqrt(1 + x * x)
        g = mp.sqrt((a + b) * (a - b))
        e = b * x - a * x * x / (sq + 1) - b * b / (a + g)
        r = mp.log(a / mp.pi) + logk1e(a * sq) - mp.log(sq) + e
    return +r


def pdf(x, a, b):
    return mp.exp(logpdf(x, a, b))


def _logerfc(z):
    # mp.erfc raises OverflowError for |z| ~ 1e150 and beyond
    if z > 1e25:
        return -z * z - mp.log(z * mp.sqrt(mp.pi)) + mp.log1p(-1 / (2 * z * z))
    if z < -20:
        return mp.log(2)
    return mp.log(mp.erfc(z))


def _L(s, x, a, b, g, sgn):
    # log of the mixture integrand in s = log v (includes the Jacobian v)
    v = mp.exp(s)
    t = sgn * (x - b * v) / mp.sqrt(v)
    # gamma - 1/(2v) - gamma^2 v/2 = -(gamma v - 1)^2/(2v), written without cancellation
    return (-mp.log(2 * mp.pi) / 2 - mp.log(2) - s / 2 - (g * v - 1) ** 2 / (2 * v)
            + _logerfc(-t / mp.sqrt(2)))


def _peak(f, s0, extra=0):
    # extra > 0 (gamma > 1e10): the peak in s is ~gamma^-1/2 wide; refine further
    with mp.workdps(30 + extra):
        ks = list(range(-40, 41))
        vals = [f(s0 + k) for k in ks]
        i = max(range(len(ks)), key=lambda j: vals[j])
        lo, hi = s0 + ks[i] - 1, s0 + ks[i] + 1
        phi = (mp.sqrt(5) - 1) / 2
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = f(c), f(d)
        for _ in range(70 + int(2.4 * extra)):
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = f(d)
        sp = (lo + hi) / 2
        Lp = f(sp)
        h = mp.mpf(10) ** (-4 - extra // 2)
        while True:
            d2 = (f(sp + h) - 2 * Lp + f(sp - h)) / h ** 2
            if d2 < 0 or h > 1:
                break
            h *= 10
        w = min(1 / mp.sqrt(-d2), mp.mpf(1)) if d2 < 0 else mp.mpf(1)
    return +sp, +w, +Lp


def _mix_tail(x, a, b, sgn, hfac=20):
    # Trapezoid with step w/hfac, halved until two successive results agree
    # to 1e-(QDPS - 10) relative. A fixed w/20 is not enough when the Phi
    # factor cuts the integrand off much faster than the IG peak width w
    # (e.g. a = 10, b = 9.99999: cdf(67.4459) was 1.6e-8 too large).
    prev = _mix_tail1(x, a, b, sgn, hfac)
    while True:
        hfac *= 2
        cur = _mix_tail1(x, a, b, sgn, hfac)
        with mp.workdps(QDPS):
            if abs(cur - prev) <= mp.mpf(10) ** -(QDPS - 10) * abs(cur) or hfac >= 20 * 2 ** 8:
                return cur
        prev = cur


def _mix_tail1(x, a, b, sgn, hfac=20):
    # sgn = 1: cdf, sgn = -1: sf
    g = mp.sqrt((a + b) * (a - b))
    # huge gamma: v ~ 1/gamma is gamma^-1/2 narrow and (gamma v - 1)^2 cancels
    extra = int(mp.log10(g)) if g > 1e10 else 0
    with mp.workdps(QDPS + extra):
        x, a, b = +x, +a, +b
        g = mp.sqrt((a + b) * (a - b))
        f = lambda s: _L(s, x, a, b, g, sgn)
        sp, w, Lp = _peak(f, -mp.log(g), extra)
        if Lp < -1000:
            # far outside the double range: leading Laplace term (deterministic)
            return mp.exp(Lp) * w * mp.sqrt(2 * mp.pi)
        Lp = f(sp)
        h = w / hfac
        tol = mp.mpf(10) ** -70
        tot = mp.mpf(1)
        for sg in (1, -1):
            k = 1
            while True:
                fv = mp.exp(f(sp + sg * k * h) - Lp)
                tot += fv
                if fv < tol:
                    break
                k += 1
        return tot * h * mp.exp(Lp)


@functools.lru_cache(maxsize=4096)
def _tail0(x, a, b):
    # (cdf, sf) at QDPS-digit inputs, computed at fixed internal precision
    g = mp.sqrt((a + b) * (a - b))
    if x < b / g:
        c = _mix_tail(x, a, b, 1)
        return c, 1 - c
    s = _mix_tail(x, a, b, -1)
    return 1 - s, s


def _round(*v):
    with mp.workdps(QDPS):
        return tuple(+t for t in v)


def _tail(x, a, b):
    # The quadrature runs at a fixed precision (QDPS digits) on inputs rounded
    # to QDPS digits (float inputs are exact). The harness differentiates the
    # references with steps ~10^-(dps+3) (mp.diff); a first-order correction
    # pdf(x) * (x - x0) keeps cdf/sf differentiable in x at full working
    # precision (the neglected term is O((x - x0)^2)).
    x0, a0, b0 = _round(x, a, b)
    c, s = _tail0(x0, a0, b0)
    if x != x0:
        d = pdf(x, a, b) * (x - x0)
        c, s = c + d, s - d
    return c, s


def cdf(x, a, b):
    return +_tail(x, a, b)[0]


def sf(x, a, b):
    return +_tail(x, a, b)[1]


@functools.lru_cache(maxsize=1024)
def _isf0(q, a, b):
    # root of sf(x) = q at fixed precision QDPS (inputs already rounded):
    # safeguarded Newton in u = asinh(x) on a decreasing function G(u).
    with mp.workdps(QDPS):
        upper = q > mp.mpf(0.5)
        if upper:
            # 1 - q is exact here (q has at most QDPS digits, 1 - q < 1/2)
            with mp.workdps(2 * QDPS + 20):
                lq = mp.log(1 - q)
            G = lambda u: lq - mp.log(cdf(mp.sinh(u), a, b))
            dG = lambda u: -pdf(mp.sinh(u), a, b) * mp.cosh(u) / cdf(mp.sinh(u), a, b)
        else:
            lq = mp.log(q)
            G = lambda u: mp.log(sf(mp.sinh(u), a, b)) - lq
            dG = lambda u: -pdf(mp.sinh(u), a, b) * mp.cosh(u) / sf(mp.sinh(u), a, b)
        g = mp.sqrt((a + b) * (a - b))
        u0 = mp.asinh(b / g)
        lo = hi = u0
        step = mp.mpf(1)
        if G(u0) > 0:
            while True:
                hi = u0 + step
                if G(hi) < 0:
                    break
                lo = hi
                step *= 2
        else:
            while True:
                lo = u0 - step
                if G(lo) > 0:
                    break
                hi = lo
                step *= 2
        u = (lo + hi) / 2
        for _ in range(100):
            gu = G(u)
            if gu == 0:
                break
            if gu > 0:
                lo = u
            else:
                hi = u
            un = u - gu / dG(u)
            if not (lo <= un <= hi):
                un = (lo + hi) / 2
            if abs(un - u) < mp.mpf(10) ** -(QDPS - 20) * (1 + abs(u)):
                u = un
                break
            u = un
        return mp.sinh(u)


def isf(q, a, b):
    # fixed-precision root at the rounded q0, plus the first-order correction
    # (q0 - q) / pdf(x0) so that mp.diff sees the true derivative.
    q0, a0, b0 = _round(q, a, b)
    x0 = _isf0(q0, a0, b0)
    if q != q0:
        x0 = x0 + (q0 - q) / pdf(x0, a, b)
    return +x0


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a, b: mp.log(cdf(x, a, b)),
    "sf": sf,
    "logsf": lambda x, a, b: mp.log(sf(x, a, b)),
    "isf": isf,
}
METHODS = ["pdf", "cdf", "sf", "isf"]


def x_points(a, b):
    import numpy as np
    g = np.sqrt((a + b) * (a - b))
    mean = b / g
    sd = a / g ** 1.5
    pts = [mean, -mean]
    pts += [mean + k * sd for k in (-30, -10, -5, -3, -1, -0.3, 0.3, 1, 3, 5, 10, 30)]
    pts += [k / np.sqrt(a) for k in (0.1, 1, 3, -1, -3)]
    # where the exponent (a -+ b) |x| is ~ 100 and ~ 600 (deep tails)
    for s in (1, -1):
        d = a - s * b
        pts += [s * 100 / d, s * 600 / d]
    return np.array([p for p in pts if np.isfinite(p)])


def q_points(a, b):
    # isf results close to 0 (brentq xtol = 1e-14 is absolute): q near sf(0)
    import numpy as np
    from scipy import stats
    s0 = float(stats.norminvgauss.sf(0.0, a, b))
    return np.array([s0 * (1 + r) for r in (-1e-3, -1e-6, -1e-9, 1e-9, 1e-6, 1e-3)
                     if 0 < s0 * (1 + r) < 1])
