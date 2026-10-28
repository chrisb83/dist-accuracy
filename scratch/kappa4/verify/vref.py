"""Independent mpmath reference for scipy.stats.kappa4 (standard form).

Derivation, from the docstring pdf
    f(x,h,k) = (1-k x)^(1/k-1) * (1 - h (1-k x)^(1/k))^(1/h-1).
Write  ly = log(1-k x)/k   (so y = (1-k x)^(1/k) = exp(ly); ly = -x for k=0).
Then
    log F = log1p(-h*exp(ly))/h            (h != 0)
          = -exp(ly)                       (h  = 0)        [limit]
    log f = (1-k)*ly + (1-h)*log F         (all branches)
because (1/k-1)*log(1-k x) = (1-k)*ly and (1/h-1)*log1p(-h y) = (1-h)*log F.
This route never forms y^k or (1-h y)^(1/h) and is a different arrangement
from the implementation (and from a direct textbook transcription).

Overflow handling, with explicit truncation bound instead of a saturating exp:
let L = log|h| + ly = log|h y|.
  h < 0:  log1p(-h y) = log1p(exp(L)) = L + log1p(exp(-L));
          for L > 1e4 the omitted term is < exp(-1e4) < 1e-4342, far below the
          200-digit working accuracy, so it is dropped exactly.
  h > 0:  log1p(-h y) = log(-expm1(L)), accurate up to the support endpoint
          where L -> 0.
ppf: y = -expm1(h log q)/h (both signs of h), ly_q = log|expm1(h log q)| - log|h|
     computed as (h log q) + log1p(-exp(-h log q)) - log|h| when h log q > 1,
     then x = -expm1(k*ly_q)/k  (x = -ly_q for k=0).
"""
from mpmath import mp

BIG = mp.mpf('1e4')


def _ly(x, k):
    x = mp.mpf(x)
    k = mp.mpf(k)
    if k == 0:
        return -x
    return mp.log1p(-k*x)/k


def _log1p_mhy(x, h, k):
    """log(1 - h*y) for h != 0."""
    h = mp.mpf(h)
    ly = _ly(x, k)
    L = mp.log(abs(h)) + ly
    if h < 0:
        if L > BIG:
            return L
        if L < -BIG:
            return mp.exp(L)          # log1p(small)
        return mp.log1p(mp.exp(L))
    else:
        if L > 0:
            raise ValueError('outside support (h*y > 1)')
        return mp.log(-mp.expm1(L))


def logcdf(x, h, k):
    h = mp.mpf(h)
    if h == 0:
        ly = _ly(x, k)
        if ly > BIG:
            raise ValueError('cdf underflows below any double')
        return -mp.exp(ly)
    return _log1p_mhy(x, h, k)/h


def cdf(x, h, k):
    return mp.e**logcdf(x, h, k)


def logpdf(x, h, k):
    h = mp.mpf(h)
    k = mp.mpf(k)
    return (1-k)*_ly(x, k) + (1-h)*logcdf(x, h, k)


def pdf(x, h, k):
    return mp.e**logpdf(x, h, k)


def _ly_of_q(q, h):
    """log y where y = (1-q**h)/h  (h != 0), or y = -log q (h == 0)."""
    q = mp.mpf(q)
    h = mp.mpf(h)
    if h == 0:
        return mp.log(-mp.log(q))
    A = h*mp.log(q)
    if A > BIG:
        lnum = A
    elif A > 0:
        # log(exp(A)-1) = A + log1p(-exp(-A))
        lnum = A + mp.log1p(-mp.exp(-A))
    else:
        # |expm1(A)| = 1 - exp(A); log1p keeps relative accuracy for A << 0
        lnum = mp.log1p(-mp.exp(A))
    return lnum - mp.log(abs(h))


def ppf(q, h, k):
    k = mp.mpf(k)
    ly = _ly_of_q(q, h)
    if k == 0:
        return -ly
    return -mp.expm1(k*ly)/k


def support_lower(h, k):
    h = mp.mpf(h)
    k = mp.mpf(k)
    if h > 0 and k != 0:
        # x where h*y = 1: 1 - k x = h**(-k)  ->  x = (1 - h**-k)/k
        return -mp.expm1(-k*mp.log(h))/k
    if h > 0 and k == 0:
        return mp.log(h)
    if k < 0:
        return 1/k
    return mp.mpf('-inf')
