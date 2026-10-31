# Verifier reference for k = 2: Q = sqrt(2)|N|/S, S = chi_df/sqrt(df).
# Route A: integral over u = log s of f_S(e^u) e^u * conditional cdf/pdf (mp.quad).
# Route B: Student-t cdf via hyp2f1 (cdf) / closed-form t pdf (pdf check).
import mpmath as mp

def logfS(s, df):
    return (df/2)*mp.log(df) + (df-1)*mp.log(s) - df*s*s/2 - (df/2-1)*mp.log(2) - mp.loggamma(df/2)

def breaks(x, df):
    c = -mp.log(x)  # s ~ 1/x
    pts = set()
    for k in range(-60, 12, 2): pts.add(mp.mpf(k))
    for k in range(-80, 12, 2): pts.add(c + k)
    w = 1/mp.sqrt(2*df)
    for k in range(-40, 41): pts.add(k*w/4)
    for k in range(-40, 41): pts.add(c + k*w/4) if df > 1000 else None
    lo = min(c - 90, mp.mpf(-70) - 80/ (1+df))
    pts = sorted(p for p in pts if lo < p < 15)
    return [lo] + pts + [mp.mpf(15)]

def cdfA(x, df):
    x = mp.mpf(x); df = mp.mpf(df)
    f = lambda u: mp.exp(logfS(mp.exp(u), df) + u) * mp.erf(x*mp.exp(u)/2)
    return mp.quad(f, breaks(x, df))

def pdfA(x, df):
    x = mp.mpf(x); df = mp.mpf(df)
    f = lambda u: mp.exp(logfS(mp.exp(u), df) + 2*u - x*x*mp.exp(2*u)/4) * mp.sqrt(2)/mp.sqrt(2*mp.pi)
    return mp.quad(f, breaks(x, df))

def cdfB(x, df):
    x = mp.mpf(x); nu = mp.mpf(df); t = x/mp.sqrt(2)
    return 2*t*mp.gamma((nu+1)/2)/(mp.sqrt(mp.pi*nu)*mp.gamma(nu/2))*mp.hyp2f1(0.5, (nu+1)/2, 1.5, -t*t/nu)

def pdfB(x, df):
    x = mp.mpf(x); nu = mp.mpf(df); t = x/mp.sqrt(2)
    return mp.sqrt(2)*mp.gamma((nu+1)/2)/(mp.sqrt(mp.pi*nu)*mp.gamma(nu/2))*(1+t*t/nu)**(-(nu+1)/2)
