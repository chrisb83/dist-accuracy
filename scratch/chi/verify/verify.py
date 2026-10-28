"""Independent verifier references for chi (standard form).
pdf/logpdf: chi2 relation pdf_chi(x) = 2x * pdf_chi2(x^2; df).
cdf: small df -> y^a/Gamma(a+1) * 1F1(a; a+1; -y)  (a=df/2, y=x^2/2);
     large df -> mp.quad of the docstring pdf over split intervals.
sf = 1 - cdf. ppf/isf: secant/Newton root of my cdf, checked cdf(x)=q.
"""
import sys, time
import mpmath as mp
import numpy as np
from scipy import stats

def logpdf(x, df):
    x = mp.mpf(x); df = mp.mpf(df); k2 = df/2
    y = x*x
    # log chi2 pdf at y
    lchi2 = (k2-1)*mp.log(y) - y/2 - k2*mp.log(2) - mp.loggamma(k2)
    return mp.log(2*x) + lchi2

def pdf(x, df):
    return mp.exp(logpdf(x, df))

def cdf_series(x, df):
    x = mp.mpf(x); a = mp.mpf(df)/2; y = x*x/2
    return mp.exp(a*mp.log(y) - mp.loggamma(a+1)) * mp.hyp1f1(a, a+1, -y)

def cdf_quad(x, df):
    x = mp.mpf(x)
    pts = [mp.mpf(0)] + [x - d for d in (400, 100, 40, 15, 5, 2, 1, mp.mpf('0.3')) if x - d > 0] + [x]
    return mp.quad(lambda t: pdf(t, df), pts)

def cdf(x, df):
    return cdf_series(x, df) if df <= 3 else cdf_quad(x, df)

def quantile(target_cdf, df, x0):
    # solve cdf(x) = target in log-x for tiny values
    if df <= 3:
        a = mp.mpf(df)/2
        # leading term start
        u0 = (mp.log(target_cdf) + mp.loggamma(a+1) + a*mp.log(2))/df
        f = lambda u: mp.log(cdf_series(mp.exp(u), df)) - mp.log(target_cdf)
        u = mp.findroot(f, u0)
        return mp.exp(u)
    x = mp.mpf(x0)
    for _ in range(8):
        dx = (cdf_quad(x, df) - target_cdf)/pdf(x, df)
        x -= dx
        if abs(dx) < abs(x)*mp.mpf(10)**(-mp.mp.dps+10):
            break
    return x

def ref(method, arg, df, sx):
    if method == 'pdf': return pdf(arg, df)
    if method == 'logpdf': return logpdf(arg, df)
    if method == 'cdf': return cdf(arg, df)
    if method == 'sf': return 1 - cdf(arg, df)
    q = mp.mpf(arg)
    if method == 'ppf': x = quantile(q, df, sx)
    else: x = quantile(1 - q, df, sx)
    # check
    c = cdf(x, df) if method == 'ppf' else 1 - cdf(x, df)
    assert abs(c - q) <= abs(q)*mp.mpf(10)**(-40), (method, arg, df, c, q)
    return x

CASES = [l.split() for l in """
F1 cdf 1e-200 1
F1 cdf 1e-200 0.001
F1 cdf 1e-300 0.1
F1 cdf 1e-160 1
F1 cdf 1e-160 0.1
F1 sf 1e-200 0.001
F1 sf 1e-160 0.001
F2 ppf 1e-200 1
F2 ppf 0.57 0.001
F2 ppf 0.5000000000000001 0.001
F2 ppf 1e-17 0.1
F2 ppf 1e-16 0.1
F2 isf 0.36 0.001
F2 isf 0.42999999999999994 0.001
F3 logpdf 1.35e154 78
F3 logpdf 1.5e154 78
F3 logpdf 1.8e154 2e6
F4 cdf 1410.9571219565817 2e6
F4 cdf 1410.6735979665884 2e6
F4 sf 1410.9571219565817 2e6
F4 cdf 14138.882558391946 2e8
F4 cdf 14135.062787267696 2e8
F4 cdf 14120.906486483083 2e8
F4 sf 14138.882558391946 2e8
F5 pdf 14140.014144264496 2e8
F5 pdf 14142.13562373095 2e8
F5 pdf 1419.1546779685434 2e6
F5 pdf 1414.213562373095 2e6
F5 pdf 31.622776601683793 1000
F6 logpdf 14142.13562373095 2e8
F6 logpdf 1414.2132088196602 2e6
F6 logpdf 1414.213562373095 2e6
F6 logpdf 31.622776601683793 1000
F7 isf 0.9999993371296838 2e6
F7 isf 0.999999938945977 2e6
F7 ppf 1e-08 2e6
F8 ppf 0.9 0.001
F8 ppf 0.8499999999999999 0.001
F8 ppf 0.7799999999999999 0.001
""".strip().splitlines()]

only = sys.argv[1:]
import warnings; warnings.simplefilter('ignore')
for fid, meth, arg, df in CASES:
    if only and fid not in only: continue
    a = float(arg); d = float(df)
    s = float(getattr(stats.chi, meth)(a, d))
    t0 = time.time()
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(ref(meth, mp.mpf(a), d, s if s > 0 else 1.0))
    mp.mp.dps = 200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    r = vals[1]
    ulp = abs(mp.mpf(s) - r)/(abs(r)*mp.mpf(2)**-52) if r != 0 else mp.nan
    print(f"{fid} {meth}({arg}, {df}) scipy={s!r} ref={mp.nstr(r,17)} ulp={mp.nstr(ulp,3)} agree30={agree} t={time.time()-t0:.1f}s", flush=True)
