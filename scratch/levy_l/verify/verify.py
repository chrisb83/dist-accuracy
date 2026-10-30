import mpmath as mp, numpy as np
from scipy import stats

def phi(t): return mp.exp(-t*t/2)/mp.sqrt(2*mp.pi)
def cdf_q(x):  # substitution t = |y|^-1/2: cdf = 2*int_0^z phi
    z = 1/mp.sqrt(-x); return 2*mp.quad(phi, [0, z])
def sf_q(x):
    # 2*int_z^inf phi(t) dt with t = z + v/z: (2/z) phi(z) int_0^inf exp(-v - v^2/(2 z^2)) dv
    z = 1/mp.sqrt(-x)
    return 2*phi(z)/z*mp.quad(lambda v: mp.exp(-v - v*v/(2*z*z)), [0, 1, 10, 100, mp.inf])
def cdf_series(x):
    z = 1/mp.sqrt(-x); s = 0; n = 0
    while True:
        term = (-1)**n * z**(2*n+1)/(2**n*mp.factorial(n)*(2*n+1))
        s += term; n += 1
        if abs(term) < abs(s)*mp.eps: break
    return mp.sqrt(2/mp.pi)*s
def pdf(y):
    a=-y; return 1/(a*mp.sqrt(2*mp.pi*a))*mp.exp(-1/(2*a))
def cdf_direct(x):  # quad of the docstring pdf in x
    return mp.quad(pdf, [-mp.inf, 100*x, 10*x, x])
def ppf_ref(q):
    # Newton on cdf_q(x) - q in variable z, start from series z ~ q*sqrt(pi/2)
    q = mp.mpf(q)
    f = lambda z: (2*mp.quad(phi,[0,z]) - q)/q
    z = mp.findroot(f, q*mp.sqrt(mp.pi/2) if q < 0.5 else mp.mpf(1))
    return -1/z**2

cases = [('cdf',-1.7e308),('cdf',-1e32),('cdf',-1e31),('cdf',-1e20),('cdf',-1e16),
         ('cdf',-1e10),('cdf',-1e8),('cdf',-1e6),
         ('ppf',1e-100),('ppf',1e-16),('ppf',1e-15),('ppf',1e-12),('ppf',1e-8),('ppf',1e-6),('ppf',1e-3),
         ('sf',-0.00071),('sf',-0.001)]
for m, a in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        x = mp.mpf(a)
        if m=='cdf': v = cdf_series(x) if a < -1e300 else cdf_q(x)
        elif m=='sf': v = sf_q(x)
        else: v = ppf_ref(x)
        vals.append(v)
    mp.mp.dps = 200
    agree = abs(vals[0]-vals[1])/abs(vals[1])
    ref = vals[1]
    extra = ''
    if m=='ppf':
        extra = 'cdf(xref)/q-1=%s' % mp.nstr(cdf_q(ref)/mp.mpf(a)-1, 3)
    if m=='cdf' and a >= -1e20:
        mp.mp.dps = 50
        extra = 'direct-quad rel diff=%s' % mp.nstr(cdf_direct(mp.mpf(a))/ref-1, 3)
        mp.mp.dps = 200
    s = getattr(stats.levy_l, m)(a)
    r = float(ref)
    ulp = abs(s - mp.mpf(r) if False else mp.mpf(float(s)) - ref)/abs(ref)/mp.mpf(2.220446049250313e-16) if np.isfinite(s) else mp.inf
    print(f"{m}({a!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={mp.nstr(ulp,3)} agree100v200={mp.nstr(agree,3)} {extra}")
