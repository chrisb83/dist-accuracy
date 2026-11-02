import numpy as np, mpmath as mp, warnings
from scipy import stats
warnings.simplefilter("ignore")

def pdf(x, b, c):
    x, b, c = mp.mpf(x), mp.mpf(b), mp.mpf(c)
    return b / (1 - c**(-b)) * x**(-(b+1))

# Route: expm1 identities (not the 1 - x**-b form), checked by d/dx cdf == pdf below.
def cdf(x, b, c):
    x, b, c = mp.mpf(x), mp.mpf(b), mp.mpf(c)
    return mp.expm1(-b*mp.log(x)) / mp.expm1(-b*mp.log(c))
def sf(x, b, c):
    x, b, c = mp.mpf(x), mp.mpf(b), mp.mpf(c)
    return mp.exp(-b*mp.log(c)) * mp.expm1(b*(mp.log(c) - mp.log(x))) / -mp.expm1(-b*mp.log(c))
def ppf(q, b, c):
    q, b, c = mp.mpf(q), mp.mpf(b), mp.mpf(c)
    x = (1 - q*(1 - c**(-b)))**(-1/b)
    return x, cdf(x, b, c) - q   # residual check via quad
def isf(q, b, c):
    q, b, c = mp.mpf(q), mp.mpf(b), mp.mpf(c)
    x = (c**(-b) + q*(1 - c**(-b)))**(-1/b)
    return x, sf(x, b, c) - q

REF = {
 'pdf': lambda x,b,c: pdf(x,b,c),
 'logpdf': lambda x,b,c: mp.log(pdf(x,b,c)),
 'cdf': cdf, 'sf': sf,
 'logcdf': lambda x,b,c: mp.log(cdf(x,b,c)),
 'logsf': lambda x,b,c: mp.log1p(-cdf(x,b,c)) if cdf(x,b,c) < sf(x,b,c) else mp.log(sf(x,b,c)),
 'ppf': lambda q,b,c: ppf(q,b,c)[0],
 'isf': lambda q,b,c: isf(q,b,c)[0],
}

CASES = [
 ('F1','pdf',5,-1000,5),('F1','pdf',4.99,-440.5,5),('F1','logpdf',1,-1000,5),
 ('F1','cdf',3,-1000,5),('F1','logcdf',1.000000000001,-1000,5),('F1','sf',4.9999999996,-1000,5),
 ('F1','ppf',0.5,-1000,5),('F1','isf',0.5,-1000,5),('F1','pdf',1e100,-2,1e200),
 ('F1','logpdf',1,-2,1e200),('F1','cdf',1e100,-2,1e200),('F1','logsf',1e100,-2,1e200),
 ('F1','ppf',0.5,-2,1e200),
 ('F2','cdf',1.000000000001,1e-8,5),('F2','logcdf',1.000000000001,1e-8,5),
 ('F2','logsf',1.000000000001,1e-8,5),('F2','sf',4.9999999996,1e-8,5),
 ('F2','logsf',4.9999999996,1e-8,5),('F2','logcdf',4.9999999996,1e-8,5),
 ('F2','logcdf',1.000000000001,-1e-8,5),
 ('F3','logsf',2.1064414349807272,1000,5),('F3','logsf',2.2,1000,5),('F3','logsf',2.4000000000000004,1000,5),
 ('F4','pdf',1,1e-8,5),('F4','pdf',1,-1e-8,5),('F4','pdf',1,2,1.00000001),
 ('F4','logpdf',1,-1e-8,5),('F4','cdf',4.999999949999999,1e-8,5),('F4','sf',1.00000001,1e-8,5),
 ('F4','pdf',3,1e-5,5),('F4','pdf',1.00005,2,1.0001),
 ('F5','ppf',0.01,1e-8,5),('F5','ppf',0.5,-1e-8,5),('F5','isf',1e-300,1e-8,5),('F5','ppf',0.5,0.001,5),
]

for fid, m, a, b, c in CASES:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(REF[m](a, b, c))
    mp.mp.dps = 200
    agree = (vals[0] == vals[1]) or abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref = vals[1]
    s = getattr(stats.truncpareto, m)(a, b, c)
    if np.isfinite(s) and ref != 0:
        ulp = float(abs(mp.mpf(float(s)) - ref) / abs(ref) / mp.mpf(2)**-52)
    else:
        ulp = float('inf')
    extra = ''
    if m in ('ppf','isf'):
        mp.mp.dps = 100
        r = (ppf if m=='ppf' else isf)(a,b,c)[1]
        extra = f' resid={mp.nstr(r,3)}'
    if m in ('cdf','sf','logcdf','logsf'):
        mp.mp.dps = 100
        F = cdf if m in ('cdf','logcdf') else (lambda x,b,c: -sf(x,b,c))
        d = mp.diff(lambda t: F(t,b,c), mp.mpf(a), h=mp.mpf(a)*mp.mpf(10)**-40)
        extra = ' dcdf/pdf-1=' + mp.nstr(d/pdf(a,b,c)-1, 3)
        mp.mp.dps = 200
        extra += ' cdf+sf-1=' + mp.nstr(cdf(a,b,c)+sf(a,b,c)-1, 3)
    print(f"{fid} {m}({a!r}, {b}, {c}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree30={agree}{extra}")
