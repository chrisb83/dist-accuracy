# Independent verifier reference for halfgennorm (standard form).
# cdf = x * exp(-y) * 1F1(1; a+1; y) / Gamma(a+1), y = x**beta, a = 1/beta (Kummer form;
#   note y**a = x exactly), cross-checked with mp.quad of the docstring pdf where feasible.
# sf = 1 - cdf. Quantiles via findroot on log-cdf / log-sf, then check cdf(x_ref) == q.
# logpdf = log(beta) - loggamma(1/beta) - x**beta.
import mpmath as mp, numpy as np, scipy, sys
from scipy import stats

def cdf(x, b):
    x = mp.mpf(x); b = mp.mpf(b); a = 1/b; y = x**b
    return x*mp.exp(-y)*mp.hyp1f1(1, a+1, y)/mp.gamma(a+1)

def cdf_quad(x, b):
    x = mp.mpf(x); b = mp.mpf(b)
    f = lambda t: b/mp.gamma(1/b)*mp.exp(-t**b)
    return mp.quad(f, [0, x/2, x])

def sf(x, b): return 1 - cdf(x, b)
def logpdf(x, b):
    x = mp.mpf(x); b = mp.mpf(b)
    return mp.log(b) - mp.loggamma(1/b) - x**b

def ppf(q, b, upper=False):
    q = mp.mpf(q)
    tgt = 1 - q if upper else q
    # initial guess: small-y leading term x ~ q*Gamma(1+a), else 0.5-ish
    a = 1/mp.mpf(b)
    x0 = min(tgt*mp.gamma(1+a), mp.mpf('0.999'))
    g = lambda lx: mp.log(cdf(mp.exp(lx), b)) - mp.log(tgt)
    lx = mp.findroot(g, mp.log(x0), tol=mp.mpf(10)**(-mp.mp.dps+10))
    x = mp.exp(lx)
    return x

CASES = [
 ('cdf',1e-300,2),('cdf',1e-300,20),('cdf',0.4,1000),('cdf',0.99,1e6),('cdf',0.5,1e300),
 ('sf',0.99,1e6),('sf',0.5,1e300),('sf',0.4,1000),('sf',1e-08,1000),('cdf',1e-160,2),
 ('cdf',1.3998449603822993e-16,20),('cdf',0.48186080160065803,1000),
 ('ppf',1e-300,2),('ppf',1e-17,20),('ppf',0.3,1000),('isf',0.7,1000),('ppf',0.7,1e6),
 ('isf',0.3,1e6),('ppf',0.9,1e300),('ppf',1e-160,2),('ppf',0.48,1000),
 ('logpdf',0.5,1e300),('logpdf',0.5,1e6),('logpdf',0.99,1000),('logpdf',0.5,1000),('logpdf',0.5,100),
]

def ref(m, arg, b):
    if m == 'cdf': return cdf(arg, b)
    if m == 'sf': return sf(arg, b)
    if m == 'logpdf': return logpdf(arg, b)
    if m == 'ppf': return ppf(arg, b)
    if m == 'isf': return ppf(arg, b, upper=True)

print('scipy', scipy.__version__)
for m, arg, b in CASES:
    mp.mp.dps = 100; r1 = ref(m, arg, b)
    mp.mp.dps = 200; r2 = ref(m, arg, b)
    agree = abs(r1 - r2) <= abs(r2)*mp.mpf(10)**-30
    extra = ''
    if m in ('ppf', 'isf'):
        c = cdf(r2, b); tgt = mp.mpf(arg) if m == 'ppf' else 1 - mp.mpf(arg)
        extra = 'invchk=%s' % mp.nstr(abs(c - tgt)/tgt, 3)
    if m in ('cdf', 'sf') and b <= 1000 and arg >= 1e-20:
        mp.mp.dps = 50
        cq = cdf_quad(arg, b); cq = cq if m == 'cdf' else 1 - cq
        extra = 'quad_rel=%s' % mp.nstr(abs(cq - r2)/abs(r2), 3)
        mp.mp.dps = 200
    with np.errstate(all='ignore'):
        s = getattr(stats.halfgennorm, m)(arg, b)
    rf = float(r2)
    ulp = abs(s - rf)/(abs(rf)*2.220446049250313e-16) if rf != 0 else float('nan')
    print(f'{m}({arg!r}, {b!r}) scipy={s!r} ref={rf!r} ulp={ulp:.3g} agree30={agree} {extra}')
