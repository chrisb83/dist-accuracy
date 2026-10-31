import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter('ignore')
D = stats.powerlognorm

def logPhi(w):  # log of standard normal cdf at w (independent route: erfc / log1p)
    if w < -1e10:  # asymptotic series; next term 1/w**2 < 1e-20 relative... use 3 terms
        u = 1/w**2
        return -w**2/2 - mp.log(-w) - mp.log(mp.sqrt(2*mp.pi)) + mp.log(1 - u + 3*u**2 - 15*u**3 + 105*u**4)
    if w > 0:
        return mp.log1p(-mp.erfc(w/mp.sqrt(2))/2)
    return mp.log(mp.erfc(-w/mp.sqrt(2))/2)

def logpdf(x, c, s):
    x, c, s = mp.mpf(x), mp.mpf(c), mp.mpf(s)
    z = mp.log(x)/s
    return mp.log(c) - mp.log(x) - mp.log(s) - z**2/2 - mp.log(mp.sqrt(2*mp.pi)) + (c-1)*logPhi(-z)

def logsf(x, c, s):
    x, c, s = mp.mpf(x), mp.mpf(c), mp.mpf(s)
    return c*logPhi(-mp.log(x)/s)

def cdf(x, c, s):
    return -mp.expm1(logsf(x, c, s))

def cdf_quad(x, c, s):
    # integral of the docstring pdf in t = log x
    c, s = mp.mpf(c), mp.mpf(s)
    f = lambda t: c/s*mp.npdf(t/s)*mp.exp((c-1)*logPhi(-t/s))
    T = mp.log(mp.mpf(x))
    return mp.quad(f, [-mp.inf, T-40*s, T-10*s, T])

def quantile_from_logsf(L, c, s):
    # solve c*logPhi(-z) = L for z, x = exp(s z); bisection on w=-z then refine
    c, s = mp.mpf(c), mp.mpf(s)
    Lu = L/c
    g = lambda w: mp.log(-logPhi(w)) - mp.log(-Lu)
    a, b = mp.mpf(-5000), mp.mpf(60)
    for _ in range(80):
        m = (a+b)/2
        if g(m) < 0: b = m
        else: a = m
    w = mp.findroot(g, (a, b), solver='anderson')
    return mp.exp(-s*w)

def isf(q, c, s): return quantile_from_logsf(mp.log(mp.mpf(q)), c, s)
def ppf(q, c, s): return quantile_from_logsf(mp.log1p(-mp.mpf(q)), c, s)

refs = {'logpdf': logpdf, 'pdf': lambda *a: mp.exp(logpdf(*a)), 'logsf': logsf,
        'cdf': cdf, 'isf': isf, 'ppf': ppf}

cases = [
 ('F1','ppf',1e-17,2.1413923530064087,0.44639540782048337),
 ('F1','ppf',1e-17,1,1),
 ('F1','ppf',1e-300,2.1413923530064087,0.44639540782048337),
 ('F1','ppf',3.4807005884285624e-14,2.1413923530064087,0.44639540782048337),
 ('F1','ppf',1e-10,2.1413923530064087,0.44639540782048337),
 ('F1','ppf',3.4807005884285624e-14,5,0.001),
 ('F2','isf',0.1,0.001,1),
 ('F2','ppf',0.9,0.001,1),
 ('F2','isf',1e-170,0.5,1),
 ('F2','isf',1e-200,0.5,1e-152),
 ('F2','isf',0.4752,0.001,1),
 ('F3','isf',0.5,1e20,1),
 ('F3','ppf',0.5,1e20,1),
 ('F3','isf',0.9999999999999627,1000,0.5),
 ('F3','isf',0.5,1e16,1),
 ('F3','isf',0.5,1e10,1),
 ('F3','isf',0.99,1e6,1),
 ('F4','logpdf',1e-80,0.5,1e-152),
 ('F4','logpdf',1e70,0.5,1e-152),
 ('F4','logpdf',1.4e65,1,1e-152),
 ('F5','logsf',1e100,0.5,1e-152),
 ('F5','logpdf',1e100,0.5,1e-152),
 ('F5','logpdf',7.225973768125954e+86,0.5,1e-152),
 ('F6','cdf',2.104223637677623e-17,1e20,1),
 ('F6','logsf',2.104223637677623e-17,1e20,1),
 ('F6','cdf',3.1391327920480296e-17,1e10,1),
 ('F7','pdf',1e180,0.001,1),
 ('F7','pdf',1e40,0.001,1),
 ('F7','logpdf',1e180,0.001,1),
]
for fid, meth, a, c, s in cases:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(refs[meth](a, c, s))
    mp.mp.dps = 200
    agree = mp.nstr(abs(vals[0]-vals[1])/abs(vals[1]), 3) if vals[1] != 0 else 'exact0'
    ref = vals[1]
    sp = getattr(D, meth)(a, c, s)
    if np.isfinite(sp) and sp != 0 or (sp == 0 and ref == 0):
        ulp = float(abs(mp.mpf(sp)-ref)/abs(ref)/mp.mpf(2.220446049250313e-16)) if ref != 0 else 0.0
    else:
        ulp = float('inf')
    extra = ''
    if meth in ('ppf', 'isf'):
        mp.mp.dps = 200
        if meth == 'ppf':
            back = cdf(ref, c, s); tgt = mp.mpf(a)
        else:
            back = mp.exp(logsf(ref, c, s)); tgt = mp.mpf(a)
        extra = 'roundtrip rel ' + mp.nstr(abs(back-tgt)/tgt, 3)
    if meth == 'cdf' or (meth in ('logpdf',) and fid=='F7'):
        pass
    print(f"{fid} {meth}({a!r}, {c!r}, {s!r}) scipy={sp!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} 100v200={agree} {extra}")
# independent cdf check by quadrature for F6 points
for x, c, s in [(2.104223637677623e-17,1e20,1),(3.1391327920480296e-17,1e10,1)]:
    mp.mp.dps = 50
    print('quad cdf', x, c, mp.nstr(cdf_quad(x,c,s),17), mp.nstr(cdf(x,c,s),17))
