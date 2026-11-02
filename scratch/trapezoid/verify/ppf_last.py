# Last F2 reproducer: ppf(1e-300, 1e-20, 0.9). Bisection on log x, cdf by normalised quadrature (from verify.py).
import mpmath as mp, warnings
from scipy import stats
warnings.simplefilter('ignore')
import importlib.util, sys
src=open('verify.py').read().split('cases=')[0]
exec(src)
def ppf_log(q,c,d):
    q,c,d=mp.mpf(q),mp.mpf(c),mp.mpf(d)
    lo,hi=mp.log(mp.mpf('1e-200')),mp.log(c)
    for _ in range(mp.mp.prec+30):
        mid=(lo+hi)/2
        if cdf(mp.exp(mid),c,d)<q: lo=mid
        else: hi=mid
    return mp.exp((lo+hi)/2)
r=both(ppf_log,1e-300,1e-20,0.9)
s=stats.trapezoid.ppf(1e-300,1e-20,0.9)
mp.mp.dps=100
print(repr(float(s)), mp.nstr(r,17), '%.3g'%ulp(s,r), 'cdf(ref)/q-1=',mp.nstr(cdf(r,1e-20,0.9)/mp.mpf(1e-300)-1,3))
