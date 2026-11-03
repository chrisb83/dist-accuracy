# Independent reference: cdf by mp.quad of the docstring pdf; quantiles by findroot on it.
import mpmath as mp, numpy as np
from scipy import stats

def pdf(x, b):
    return b / (2*mp.gamma(1/b)) * mp.exp(-abs(x)**b)

def cdf(x, b):
    x = mp.mpf(x); b = mp.mpf(b)
    I = mp.quad(lambda t: pdf(t, b), [0, abs(x)])
    return mp.mpf(1)/2 + mp.sign(x)*I

def ppf(q, b):
    q = mp.mpf(q); b = mp.mpf(b)
    x0 = 2*q - 1
    x = mp.findroot(lambda x: cdf(x, b) - q, x0, solver='secant', tol=mp.mpf(10)**(-mp.mp.dps+5))
    return x

def ref(meth, arg, b, dps):
    with mp.workdps(dps):
        if meth == 'cdf': return cdf(arg, b)
        if meth == 'sf':  return cdf(-mp.mpf(arg), b)
        if meth == 'ppf': return ppf(arg, b)
        if meth == 'isf': return -ppf(arg, b)

cases = [('ppf',0.6,1000),('isf',0.6,1000),('ppf',0.5001,100),('ppf',0.7,1e6),
         ('ppf',0.99,1e300),('ppf',0.74,1000),
         ('cdf',-0.4,1000),('cdf',0.4,1000),('sf',-0.4,1000),('cdf',0.99,1e6),
         ('cdf',0.5,1e300),('cdf',5e-4,100),('cdf',1e-4,100)]
for meth, arg, b in cases:
    r1 = ref(meth, arg, b, 100); r2 = ref(meth, arg, b, 200)
    with mp.workdps(200):
        agree = abs(r1-r2)/abs(r2) < mp.mpf(10)**-30
        extra = ''
        if meth in ('ppf','isf'):
            qq = mp.mpf(arg) if meth=='ppf' else 1-mp.mpf(arg)
            extra = ' resid=%s' % mp.nstr(abs(cdf(r2 if meth=='ppf' else -r2, b)-qq), 3)
    s = float(getattr(stats.gennorm, meth)(arg, b))
    ulp = float(abs(mp.mpf(s)-r2)/abs(r2)/mp.mpf(2.220446049250313e-16))
    print(f"{meth}({arg}, {b}) scipy={s!r} ref={mp.nstr(r2,17)} ulp={ulp:.3g} agree30={agree}{extra}")
