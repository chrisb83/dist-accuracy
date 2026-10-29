# Independent verifier reference for scipy.stats.f (standard form).
# pdf from the docstring; cdf/sf via mp.betainc on the respective (complementary) argument;
# ppf via findroot in log x on log(cdf) - log(q), then checked: cdf(x_ref)/q - 1.
import sys, time
import mpmath as mp
import numpy as np
from scipy import stats

def logpdf(x, n, m):
    x, n, m = mp.mpf(x), mp.mpf(n), mp.mpf(m)
    # beta-prime form with t = n x / m, log1p avoids losing n x against m
    t = n*x/m
    if n == 2:  # B(1, b) = 1/b = 2/m: constants cancel exactly
        return -(1 + m/2)*mp.log1p(t)
    return (mp.log(n/m) + (n/2-1)*mp.log(t) - (n+m)/2*mp.log1p(t)
            - (mp.loggamma(n/2)+mp.loggamma(m/2)-mp.loggamma((n+m)/2)))

def pdf(x, n, m):
    return mp.exp(logpdf(x, n, m))

def cdf(x, n, m):
    x, n, m = mp.mpf(x), mp.mpf(n), mp.mpf(m)
    if n*x > m:   # y would round to 1: use the complement
        return 1 - sf(x, n, m)
    y = n*x/(m+n*x)
    return mp.betainc(n/2, m/2, 0, y, regularized=True)

def sf(x, n, m):
    x, n, m = mp.mpf(x), mp.mpf(n), mp.mpf(m)
    yc = m/(m+n*x)
    return mp.betainc(m/2, n/2, 0, yc, regularized=True)

def ppf(q, n, m):
    q, n, m = mp.mpf(q), mp.mpf(n), mp.mpf(m)
    a, b = n/2, m/2
    # leading order: I_y ~ y^a/(a B) -> y0
    ly0 = (mp.log(q) + mp.log(a) + mp.log(mp.beta(a, b)))/a
    lx0 = ly0 + mp.log(m/n)  # y ~ n x / m for small y
    f = lambda lx: mp.log(cdf(mp.exp(lx), n, m)) - mp.log(q)
    lx = mp.findroot(f, lx0, solver='secant', tol=mp.mpf(10)**(-mp.mp.dps+10))
    x = mp.exp(lx)
    return x, cdf(x, n, m)/q - 1

CASES = [
 ('F1','logpdf',1e303,1e6,0.01), ('F1','pdf',1e303,1e6,0.01),
 ('F1','logpdf',1e308,29,18), ('F1','logpdf',1e308,1e5,3e5),
 ('F2','sf',1e308,2,0.01), ('F2','sf',1e303,1e6,1), ('F2','sf',1e304,1e5,0.5), ('F2','cdf',1e308,2,0.01),
 ('F3','ppf',1e-170,5,1e4), ('F3','ppf',1e-300,5,1e4), ('F3','ppf',1e-180,10,100),
 ('F4','ppf',1e-153,1,1e4),
 ('F5','logpdf',1e-20,2,0.01), ('F5','logpdf',1e-120,2,0.01), ('F5','logpdf',1e-10,2,0.01),
 ('F6','pdf',1e120,1e6,2), ('F6','pdf',4,1e6,2), ('F6','logpdf',0.5,1e6,2),
 ('F6','pdf',0.999999999999,1e5,3e5), ('F6','logpdf',1.0103814730063743,1e5,3e5),
 ('F7','sf',1e-16,1,1), ('F7','cdf',1e20,1,1), ('F7','cdf',1e10,1,1),
 ('F8','cdf',0.9496708976337849,1e5,3e5),
]
only = sys.argv[1:] 
for fid, meth, arg, n, m in CASES:
    if only and fid not in only: continue
    t = time.time()
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        fn = globals()[meth]
        if meth == 'ppf':
            v, chk = fn(arg, n, m)
        else:
            v, chk = fn(arg, n, m), None
        vals.append((v, chk))
    mp.mp.dps = 200
    (v1, _), (v2, chk) = vals
    agree = mp.nstr(abs(v1-v2)/abs(v2), 3) if v2 != 0 else 'n/a'
    with np.errstate(all='ignore'):
        s = getattr(stats.f, meth)(arg, n, m)
    ref = float(v2)
    if np.isfinite(s) and ref != 0:
        ulp = abs(mp.mpf(float(s)) - v2)/abs(v2)/mp.mpf(2.220446049250313e-16)
        ulp = mp.nstr(ulp, 3)
    else:
        ulp = 'inf'
    extra = f" cdfcheck={mp.nstr(chk,3)}" if chk is not None else ""
    # closed form cross-check for (1,1)
    if n == 1 and m == 1:
        cf = (2/mp.pi)*(mp.atan(mp.sqrt(arg)) if meth=='cdf' else mp.atan(1/mp.sqrt(mp.mpf(arg))))
        extra += f" atan_rel={mp.nstr(abs(cf-v2)/v2,3)}"
    print(f"{fid} {meth}({arg!r}, {n!r}, {m!r}) scipy={s!r} ref={mp.nstr(v2,17)} ulp={ulp} 100vs200={agree}{extra} t={time.time()-t:.1f}s", flush=True)
