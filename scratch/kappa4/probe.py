import numpy as np, mpmath as mp
from scipy.stats import kappa4
mp.mp.dps = 60

def ref_cdf(x,h,k):
    x=mp.mpf(x); h=mp.mpf(h); k=mp.mpf(k)
    y = mp.e**(-x) if k==0 else (1-k*x)**(1/k)
    return mp.e**(-y) if h==0 else (1-h*y)**(1/h)

def ref_ppf(q,h,k):
    q=mp.mpf(q); h=mp.mpf(h); k=mp.mpf(k)
    y = -mp.log(q) if h==0 else (1-q**h)/h
    return -mp.log(y) if k==0 else (1-y**k)/k

print("support h=-1,k=0:", kappa4.support(-1,0))
for h in (-1.0, -0.5, -2.0):
    print("ppf h=%g k=0:"%h, kappa4.ppf(0.5,h,0.0), "ref", ref_ppf(0.5,h,0.0))
    print("  isf:", kappa4.isf(0.5,h,0.0))
print("cdf(0,-1,0)", kappa4.cdf(0.0,-1.0,0.0), ref_cdf(0,-1,0))
print("logistic check cdf(1,-1,0)", kappa4.cdf(1.0,-1.0,0.0), 1/(1+np.exp(-1)))
# overflow hypothesis k=0,h<0 large
for (x,h) in [(-1e8,-1e6), (-1e5,-1e3)]:
    print("cdf(%g,%g,0)"%(x,h), kappa4.cdf(x,h,0.0), "ref", mp.nstr(ref_cdf(x,h,0),8))
# overflow hypothesis k>0,h<0
print("cdf(-1e35,-1e3,0.1)", kappa4.cdf(-1e35,-1e3,0.1), "ref", mp.nstr(ref_cdf(-1e35,-1e3,0.1),8))
print("logcdf(-1e35,-1e3,0.1)", kappa4.logcdf(-1e35,-1e3,0.1), "ref", mp.nstr(mp.log(ref_cdf(-1e35,-1e3,0.1)),8))
# small k amplification
for k in (1e-4,1e-6,1e-8):
    print("cdf(1.0,0.5,%g)"%k, repr(kappa4.cdf(1.0,0.5,k)), mp.nstr(ref_cdf(1.0,0.5,k),20))
# small h ppf cancellation
for h in (1e-4,1e-8,1e-14):
    print("ppf(0.3,%g,0.5)"%h, repr(kappa4.ppf(0.3,h,0.5)), mp.nstr(ref_ppf(0.3,h,0.5),20))
