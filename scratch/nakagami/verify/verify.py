# Independent verifier references for nakagami (standard form).
# pdf: f(x) = 2 nu^nu / Gamma(nu) x^(2nu-1) exp(-nu x^2)
# Routes: logpdf via chi(k=2nu) identity; cdf via Kummer series
#   P(a,y) = y^a e^-y / Gamma(a+1) * 1F1(1; a+1; y)  (own summation loop);
# sf at far right tail via mp.quad of pdf over a finite interval;
# quantiles via secant on log cdf (own cdf).
import sys, warnings
import mpmath as mp
import numpy as np
from scipy import stats
warnings.simplefilter('ignore')

def kummer_cdf(x, nu):
    x = mp.mpf(x); nu = mp.mpf(nu)
    y = nu * x * x
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    s = mp.mpf(1); t = mp.mpf(1); k = 0
    while True:
        k += 1
        r = y / (nu + k)
        t *= r
        s += t
        if r < 1 and t < eps * s:
            break
    return mp.exp(nu * mp.log(y) - y - mp.loggamma(nu + 1)) * s

def logpdf(x, nu):
    x = mp.mpf(x); nu = mp.mpf(nu)
    k = 2 * nu; sc = mp.sqrt(k); t = sc * x
    return mp.log(sc) + (k - 1) * mp.log(t) - t * t / 2 - (k / 2 - 1) * mp.log(2) - mp.loggamma(k / 2)

def pdf(x, nu):
    return mp.exp(logpdf(x, nu))

def sf_quad(x, nu, upper):
    x = mp.mpf(x)
    pts = [x + mp.mpf(d) for d in (0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)] + [mp.mpf(upper)]
    return mp.quad(lambda u: pdf(u, nu), pts)

def quantile(target, nu, x0):
    # solve log cdf(x) = log target in t = log x
    lt = mp.log(target)
    g = lambda t: mp.log(kummer_cdf(mp.exp(t), nu)) - lt
    t0 = mp.log(mp.mpf(x0))
    t = mp.findroot(g, (t0, t0 * (1 + mp.mpf('1e-6')) if t0 != 0 else mp.mpf('1e-6')), solver='secant', tol=mp.mpf(10) ** (-2 * mp.mp.dps + 10), maxsteps=200)
    return mp.exp(t)

def lead_x0(q, nu):
    nu = mp.mpf(nu)
    return mp.exp((mp.log(q) + mp.loggamma(nu + 1)) / (2 * nu)) / mp.sqrt(nu)

def ref(method, a, nu, x0=None):
    nu = mp.mpf(nu)
    if method == 'cdf':
        return kummer_cdf(a, nu)
    if method == 'sf':
        if float(a) > 1.2 and float(nu) >= 1000:
            return sf_quad(a, nu, 3)
        return 1 - kummer_cdf(a, nu)
    if method == 'logpdf':
        return logpdf(a, nu)
    if method == 'pdf':
        return pdf(a, nu)
    if method == 'ppf':
        return quantile(mp.mpf(a), nu, x0 if x0 else lead_x0(mp.mpf(a), nu))
    if method == 'isf':
        tq = 1 - mp.mpf(a)
        return quantile(tq, nu, x0 if x0 else lead_x0(tq, nu))

CASES = [
    ('F1', 'cdf', 1e-220, 0.001), ('F1', 'cdf', 1e-200, 0.5), ('F1', 'cdf', 1e-160, 0.001),
    ('F1', 'sf', 1e-200, 0.001), ('F1', 'sf', 1e-160, 0.001),
    ('F2', 'ppf', 0.29, 0.001), ('F2', 'ppf', 1e-17, 0.05), ('F2', 'ppf', 1e-16, 0.05),
    ('F2', 'ppf', 1e-300, 0.5), ('F2', 'isf', 0.71, 0.001),
    ('F3', 'logpdf', 4e155, 0.001), ('F3', 'logpdf', 1.35e154, 0.5),
    ('F4', 'cdf', 0.9976973488989535, 1e6), ('F4', 'cdf', 0.9974968671630001, 1e6),
    ('F4', 'sf', 0.9976973488989535, 1e6),
    ('F5', 'pdf', 1.0, 1e6), ('F5', 'pdf', 1.002297361066066, 1e6),
    ('F6', 'logpdf', 0.9984988733093293, 1e6), ('F6', 'logpdf', 1.0, 1e6),
    ('F7', 'isf', 0.9999993371296838, 1e6), ('F7', 'ppf', 1e-8, 1e6),
    ('F8', 'ppf', 0.5000000000000001, 0.001),
    ('F9', 'sf', 1.3959524698393257, 1000), ('F9', 'cdf', 0.7250230003073993, 1000),
]

sel = sys.argv[1:] 
for fid, m, a, nu in CASES:
    if sel and fid not in sel:
        continue
    s = float(getattr(stats.nakagami, m)(a, nu))
    x0 = s if (m in ('ppf', 'isf') and s > 0 and float(nu) >= 1000) else None
    if m in ('ppf', 'isf') and s > 0 and float(nu) < 1000:
        x0 = None
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(ref(m, a, nu, x0))
    mp.mp.dps = 100
    agree = mp.nstr(abs(vals[0] - vals[1]) / abs(vals[1]), 3)
    r = vals[1]
    ulp = float(abs(mp.mpf(s) - r) / (abs(r) * mp.mpf(2) ** -52)) if r != 0 else float('nan')
    extra = ''
    if m in ('ppf', 'isf'):
        mp.mp.dps = 60
        tq = mp.mpf(a) if m == 'ppf' else 1 - mp.mpf(a)
        extra = ' resid(cdf(xref)/q-1)=' + mp.nstr(kummer_cdf(r, nu) / tq - 1, 3)
    print(f"{fid} {m}({a!r}, {nu!r}) scipy={s!r} ref={mp.nstr(r, 17)} ulp={ulp:.3g} relagree100v200={agree}{extra}", flush=True)
