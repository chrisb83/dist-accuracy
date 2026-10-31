import sys; sys.path.insert(0, '.')
import mpmath as mp
from scipy import stats
from vref import cdf, pdf

def ppf_newton(q, a):
    q = mp.mpf(q); a = mp.mpf(a)
    x = -mp.sqrt(2*(-mp.log(q))/(1+a*a))   # asymptotic start, own route
    for i in range(200):
        F = cdf(x, a)
        dx = (mp.log(F) - mp.log(q)) * F / pdf(x, a)
        x = x - dx
        if abs(dx) < abs(x)*mp.mpf(10)**(-mp.mp.dps+5):
            break
    return x

cases = [('ppf',1e-300,1e4),('ppf',1.2115276586286942e-26,100),('ppf',1e-300,4),('ppf',1e-20,4),
         ('isf',1e-300,-4),('ppf',1e-10,4),('ppf',1e-6,4)]
for m,q,a in cases:
    vals=[]
    aa = a if m=='ppf' else -a   # isf(q,a) = -ppf(q,-a)
    for d in (100,200):
        mp.mp.dps=d
        x = ppf_newton(q, aa)
        if m=='isf': x = -x
        vals.append(x)
    mp.mp.dps=100
    ref = vals[1]
    chk = cdf(ref if m=='ppf' else -ref, aa)/mp.mpf(q) - 1
    s = getattr(stats.skewnorm,m)(q,a)
    ulp = float(abs(mp.mpf(s)-ref)/abs(ref))/2.22e-16
    print(m,q,a,'scipy',repr(s),'ref',mp.nstr(ref,17),'ulp %.3g'%ulp,
          'rel100v200',mp.nstr(abs(vals[0]-vals[1])/abs(ref),3),'cdf(ref)/q-1',mp.nstr(chk,3), flush=True)
