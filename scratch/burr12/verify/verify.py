import mpmath as mp, numpy as np, warnings
import scipy, scipy.stats as st
warnings.simplefilter('ignore')
# Burr XII standard form: pdf = c d x^(c-1) (1+x^c)^(-d-1)
# Route: cdf via regularized incomplete beta I_{u/(1+u)}(1, d), u = x^c
#        (Beta(1,d) cdf at y=u/(1+u)); sf = I_{1/(1+u)}(d, 1) (complementary call)
def ref(method, a, c, d):
    # guard digits: the betainc argument u/(1+u) or 1/(1+u), and 1 - q,
    # need |log10(u)| (resp. |log10(q)|) extra digits to stay distinct from 1
    import math
    ex = abs(c*math.log10(a)) if method not in ('ppf','isf') else abs(math.log10(a)) + 400
    with mp.workdps(mp.mp.dps + int(ex) + 20):
        return +_ref(method, a, c, d)
def _ref(method, a, c, d):
    a, c, d = mp.mpf(a), mp.mpf(c), mp.mpf(d)
    if method in ('logpdf', 'pdf'):
        lp = mp.log(c) + mp.log(d) + (c-1)*mp.log(a) - (d+1)*mp.log(1 + a**c)
        return lp if method == 'logpdf' else mp.exp(lp)
    if method in ('cdf', 'logcdf', 'sf', 'logsf'):
        u = a**c
        if method in ('cdf', 'logcdf'):
            v = mp.betainc(1, d, 0, u/(1+u), regularized=True)
        else:
            v = mp.betainc(d, 1, 0, 1/(1+u), regularized=True)
        return mp.log(v) if method.startswith('log') else v
    if method in ('ppf', 'isf'):
        # q = cdf(x): x = ((1-q)^(-1/d) - 1)^(1/c); isf uses p = sf
        p = 1 - a if method == 'ppf' else a
        x = (p**(-1/d) - 1)**(1/c)
        # check by forward evaluation through betainc
        chk = mp.betainc(d, 1, 0, 1/(1 + x**c), regularized=True)
        assert abs(chk/p - 1) < mp.mpf(10)**-90, (method, a, chk, p)
        return x

cases = [
 ('F1','logpdf',1e40,10,4),('F1','logsf',1e40,10,4),('F1','logpdf',1e155,2,1),
 ('F1','logpdf',1.5,1e4,1e-6),('F1','pdf',1e10,100,0.001),('F1','sf',1e10,100,0.001),
 ('F1','cdf',2,1e4,1e-6),('F1','logcdf',1e10,100,0.001),
 ('F2','logcdf',1e-10,10,4),('F2','logcdf',1e-20,0.5,0.001),('F2','logcdf',1e-5,2,1),
 ('F2','logcdf',1.0,1e4,1e-6),
 ('F3','ppf',0.01,1e4,1e-6),('F3','ppf',0.9,100,0.001),('F3','isf',0.1,100,0.001),
 ('F3','isf',2.0322308024e-313,2,1),
 ('F4','logcdf',0.05011872336272723,10,1e12),('F4','logcdf',0.004641588833612776,3,1e6),
 ('F4','logcdf',0.1,10,1e12),
 ('F5','cdf',1.839921625376204e-32,10,1e12),('F5','logsf',1.839921625376204e-32,10,1e12),
 ('F5','cdf',4.6248075735263605e-105,3,1e6),
 ('F6','ppf',1e-300,10,1e12),('F6','ppf',2.0322308024e-313,2,1),
]
print('scipy', scipy.__version__)
for fid, m, a, c, d in cases:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(ref(m, a, c, d))
    mp.mp.dps = 200
    agree = vals[0] == 0 or abs(vals[0]/vals[1] - 1) < mp.mpf(10)**-30
    r = vals[1]
    s = float(getattr(st.burr12, m)(a, c, d))
    rf = float(r)
    if np.isfinite(s) and r != 0:
        ulp = float(abs(mp.mpf(s) - r) / abs(r)) / 2.220446049250313e-16
    else:
        ulp = float('inf')
    print(f"{fid} | {m}({a!r}, {c:g}, {d:g}) | {s!r} | {rf!r} | {ulp:.3g} | agree100/200={agree}")
