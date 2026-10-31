# Independent verifier reference for ncx2 (standard form).
# pdf: Poisson mixture of chi2 densities (log-space terms), cross-checked with Bessel form.
# cdf/sf: mp.quad of the Bessel pdf (different route from analyst's gammainc sums).
import mpmath as mp

def pdf_bessel(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    return mp.mpf(1)/2*mp.exp(-(lam+x)/2)*(x/lam)**((k-2)/4)*mp.besseli((k-2)/2, mp.sqrt(lam*x))

def logterm(j, x, k, lam):
    # log of Pois(j; lam/2) * chi2pdf(x; k+2j)
    a = k/2 + j
    return (-lam/2 + j*mp.log(lam/2) - mp.loggamma(j+1)
            + (a-1)*mp.log(x/2) - x/2 - mp.loggamma(a) - mp.log(2))

def pdf_pois(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    # dominant index: maximize over j by scanning
    jstar = int(mp.floor(max(0, (-k/2 + mp.sqrt(lam*x)/2))))
    lmax = logterm(jstar, x, k, lam)
    s = mp.mpf(0); j = jstar
    while True:
        t = mp.exp(logterm(j, x, k, lam) - lmax); s += t
        if t < mp.mpf(10)**(-mp.mp.dps-10) and j > jstar + 5: break
        j += 1
    j = jstar - 1
    while j >= 0:
        t = mp.exp(logterm(j, x, k, lam) - lmax); s += t
        if t < mp.mpf(10)**(-mp.mp.dps-10): break
        j -= 1
    return mp.exp(lmax)*s

def cdf_quad(x, k, lam, pts=None):
    x = mp.mpf(x)
    f = lambda t: pdf_bessel(t, k, lam)
    p = [0] + (pts or []) + [x]
    return mp.quad(f, p)

def sf_quad(x, k, lam, pts):
    f = lambda t: pdf_bessel(t, k, lam)
    return mp.quad(f, [mp.mpf(x)] + pts + [mp.inf])

def cdf_series(x, k, lam):
    # direct Poisson sum, every term by gammainc (used only for root finding)
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    s = 0; j = 0
    while True:
        t = mp.exp(-lam/2 + j*mp.log(lam/2) - mp.loggamma(j+1)) * mp.gammainc(k/2+j, 0, x/2, regularized=True)
        s += t
        if j > lam/2 + 10 and t < s*mp.mpf(10)**(-mp.mp.dps-5): break
        j += 1
    return s
