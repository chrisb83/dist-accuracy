import warnings; warnings.simplefilter("ignore")
import mpmath as mp, numpy as np
from scipy import stats

def ulp(c, r):
    if not np.isfinite(c): return float('inf')
    return abs(mp.mpf(c)-r)/abs(r)/mp.mpf(2)**-52

def ref_both(f):
    out=[]
    for d in (100,200):
        mp.mp.dps=d; out.append(f())
    mp.mp.dps=200
    assert abs(out[0]-out[1]) <= abs(out[1])*mp.mpf(10)**-30, out
    return out[1]

# F1: route = log of mpmath's npdf (exp then log), not the closed form
print("== norm.logpdf ==")
for x in [1.3407807929942596e154, 1.3407807929942597e154, 1.35e154, -1.5e154, 1.8e154, 1.89e154, 1.896e154]:
    r = ref_both(lambda: mp.log(mp.npdf(mp.mpf(x))))
    s = stats.norm.logpdf(x)
    print(repr(x), s, mp.nstr(r,17), float(ulp(s,r)))

# callers
def logphi(z): return mp.log(mp.npdf(z))
def logPhi(z): return mp.log(mp.ncdf(z))
print("== callers ==")
# alpha: x tiny, a=1
x,a = 6e-155,1.0
r = ref_both(lambda: -2*mp.log(mp.mpf(x)) + logphi(a-1/mp.mpf(x)) - logPhi(a))
s = stats.alpha.logpdf(x,a); print("alpha", x, a, "z=",a-1/x, s, mp.nstr(r,17), float(ulp(s,r)))
# powerlognorm: c=1, s tiny
x,c,sh = 0.5,1.0,4.5e-155
r = ref_both(lambda: mp.log(c)-mp.log(mp.mpf(x))-mp.log(mp.mpf(sh))+logphi(mp.log(mp.mpf(x))/sh)+(c-1)*logPhi(-mp.log(mp.mpf(x))/sh))
s = stats.powerlognorm.logpdf(x,c,sh); print("powerlognorm", x,c,sh, "z=",np.log(x)/sh, s, mp.nstr(r,17), float(ulp(s,r)))
# powernorm: c=1
x,c = 1.5e154,1.0
r = ref_both(lambda: mp.log(c)+logphi(mp.mpf(x))+(c-1)*logPhi(-mp.mpf(x)))
s = stats.powernorm.logpdf(x,c); print("powernorm", x,c, s, mp.nstr(r,17), float(ulp(s,r)))
# skewnorm a=1 and a=0
for a in (1.0,0.0):
    x=1.5e154
    r = ref_both(lambda: mp.log(2*mp.npdf(mp.mpf(x))*mp.ncdf(a*mp.mpf(x))))
    s = stats.skewnorm.logpdf(x,a); print("skewnorm", x,a, s, mp.nstr(r,17), float(ulp(s,r)))
# truncnorm a=-2e154,b=2e154 (mass ~1)
x,a,b = 1.5e154,-2e154,2e154
r = ref_both(lambda: logphi(mp.mpf(x)))  # mass = 1 - 2*Phi(-2e154) = 1 to any precision
s = stats.truncnorm.logpdf(x,a,b); print("truncnorm", x,a,b, s, mp.nstr(r,17), float(ulp(s,r)))
