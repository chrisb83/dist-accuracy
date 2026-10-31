"""Reference module for scipy.stats.studentized_range.

Standard form (loc = 0, scale = 1), shapes (k, df) as in SciPy. With
S = chi_df / sqrt(df) (density f_S) and the range W of k standard normals,
Q = W / S, so

    cdf(q; k, df) = int_0^inf f_S(s) R_k(q s) ds,
    pdf(q; k, df) = int_0^inf s f_S(s) R_k'(q s) ds,
    R_k(w)  = k        int phi(z) D(z, w)^(k-1) dz,
    R_k'(w) = k (k-1)  int phi(z) phi(z + w) D(z, w)^(k-2) dz,
    D(z, w) = Phi(z + w) - Phi(z),
    log f_S(s) = (df/2) log df - lgamma(df/2) - (df/2 - 1) log 2
                 + (df - 1) log s - df s^2 / 2.

Two evaluation paths:

* k == 2 (exact, fast, any precision): the range of two standard normals is
  |N(0, 2)|, so Q / sqrt(2) ~ |t_df|:
      cdf = I_{t^2/(df+t^2)}(1/2, df/2),  t = q / sqrt(2)
      pdf = sqrt(2) f_t(q / sqrt(2); df).
  This path is used for the full-grid harness run (PARAMS are all k = 2).

* other k (nested quadrature, slow; use only through --at with --dps 30):
  trapezoid rules in z (Gaussian-weighted, entire integrand) and in u = log s,
  both of which converge geometrically in 1/h^2, with outward marching until
  terms are negligible. D is formed from the two tails on the far side of the
  centre (erfc of positive arguments) and with extra working digits for small
  w, so it does not cancel. Each value is computed twice (step h and h/2) and
  a ValueError is raised if the two disagree beyond 10^-(dps-8).
  Typical cost at 30 digits: 5-60 s per value. For k >~ 1000 the outer u step
  is too coarse for the step-like R_k(qs) and the check rejects the value. Refused above 60 digits,
  except |k - 2| < 1e-20 (harness cond perturbation), which uses k = 2.
"""
import mpmath as mp

DIST = "studentized_range"

# k = 2 only (exact path). df: tiny, typical, large, and both sides of the
# df = 100000 switch to the infinite-df integrand in _cdf/_pdf.
PARAMS = [(2.0, 10.0), (2.0, 1.0), (2.0, 0.1), (2.0, 1000.0),
          (2.0, 99999.0), (2.0, 100000.0)]


# ----------------------------------------------------------------- k = 2

def _pdf2(x, df):
    t = x / mp.sqrt(2)
    lft = (mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2)
           - mp.log(df * mp.pi) / 2 - (df + 1) / 2 * mp.log1p(t * t / df))
    return mp.sqrt(2) * mp.exp(lft)


def _sf2(x, df):
    t2 = x * x / 2
    return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t2), regularized=True)


def _cdf2(x, df):
    # complementary form for t^2 > df: t^2/(df+t^2) would round to 1 for the
    # huge x reached by isf with tiny df, even at 400 digits
    t2 = x * x / 2
    if t2 > df:
        return 1 - _sf2(x, df)
    return mp.betainc(mp.mpf(1) / 2, df / 2, 0, t2 / (df + t2), regularized=True)


# ----------------------------------------------------------- general k

def _D(z, w):
    r2 = mp.sqrt(2)
    if z + w / 2 > 0:
        return (mp.erfc(z / r2) - mp.erfc((z + w) / r2)) / 2
    return (mp.erfc(-(z + w) / r2) - mp.erfc(-z / r2)) / 2


def _phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def _march(term, c, h, tol):
    """Trapezoid sum of term(c + j h), j in Z, marching outwards until three
    consecutive terms are below tol * |sum|."""
    s = term(c)
    for sgn in (1, -1):
        small, j = 0, 1
        while small < 3:
            t = term(c + sgn * j * h)
            s += t
            small = small + 1 if abs(t) <= tol * abs(s) else 0
            j += 1
            if j > 200000:
                raise ValueError("march did not terminate")
    return h * s


def _inner(w, k, deriv, hz, tol):
    if deriv:
        f = lambda z: _phi(z) * _phi(z + w) * _D(z, w) ** (k - 2)
        c = k * (k - 1)
    else:
        f = lambda z: _phi(z) * _D(z, w) ** (k - 1)
        c = k
    return c * _march(f, -w / 2, hz, tol)


def _general(x, k, df, deriv, refine):
    dps = mp.mp.dps
    tol = mp.mpf(10) ** (-dps - 5)
    # D^(k-1) peaks with width ~ 1/sqrt(k) around z = -w/2 for large k
    hz = mp.mpf('0.2') / refine / max(1, mp.sqrt(k / 4))
    sig = 1 / mp.sqrt(2 * df)
    hu = min(sig / 5, mp.mpf('0.15')) / refine
    lc = ((df / 2) * mp.log(df) - mp.loggamma(df / 2) - (df / 2 - 1) * mp.log(2))

    def term(u):
        s = mp.exp(u)
        w = x * s
        extra = int(max(0, -mp.log10(w))) + 10 if w > 0 else 10
        with mp.workdps(dps + extra):
            r = _inner(w, k, deriv, hz, tol)
        lw = lc + df * u - df * s * s / 2          # f_S(s) * s  (ds = s du)
        if deriv:
            lw += u                                # extra factor s in the pdf
        return mp.exp(lw) * r

    # centre: mode of s^df exp(-df s^2/2) is s = 1
    return _march(term, mp.mpf(0), hu, tol)


def _checked(x, k, df, deriv):
    if mp.mp.dps > 60:
        # The harness perturbs k by ~1e-100 around 2 for cond (shapes); the
        # nested quadrature is far too slow at high precision, so the k = 2
        # formula is used there (cond (shapes) then omits the k direction).
        if abs(k - 2) < mp.mpf(10) ** -20:
            return _pdf2(x, df) if deriv else _cdf2(x, df)
        raise ValueError("general k only at dps <= 60 (use --at --dps 30)")
    a = _general(x, k, df, deriv, 1)
    b = _general(x, k, df, deriv, 2)
    if abs(a - b) > abs(b) * mp.mpf(10) ** (-mp.mp.dps + 8):
        raise ValueError(f"quadrature not converged: {a} vs {b}")
    return b


# ------------------------------------------------------------- public

def pdf(x, k, df):
    if k == 2:
        return _pdf2(x, df)
    return _checked(x, k, df, True)


def cdf(x, k, df):
    if k == 2:
        return _cdf2(x, df)
    return _checked(x, k, df, False)


def sf(x, k, df):
    if k == 2:
        return _sf2(x, df)
    return 1 - cdf(x, k, df)


def ppf(q, k, df):
    from accuracy import invert
    return invert(lambda x: cdf(x, k, df), q)


def isf(q, k, df):
    from accuracy import invert
    return invert(lambda x: cdf(x, k, df), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, k, df: mp.log(pdf(x, k, df)),
    "cdf": cdf,
    "logcdf": lambda x, k, df: mp.log(cdf(x, k, df)),
    "sf": sf,
    "logsf": lambda x, k, df: mp.log(sf(x, k, df)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(k, df):
    # Left tail (cdf ~ x, below epsabs = 1e-11 for x < ~1e-11; D cancels for
    # small x), bulk, and the right tail where pdf ~ x^(-df-1) drops below 1e-11.
    # 1e-15 ... 1e-18 map where the cdf becomes exactly 0; 2e3 ... 1e8 map
    # where the quadrature stops seeing the s ~ 1/x region.
    return [1e-300, 1e-20, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-12, 1e-10,
            1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
            30.0, 100.0, 1e3, 2e3, 5e3, 1e4, 2e4, 1e5, 1e6, 1e7, 1e8]
