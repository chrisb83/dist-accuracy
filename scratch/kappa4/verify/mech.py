import sys, warnings
sys.path.insert(0,'C:/Users/chbau/Documents/research/dist-accuracy/scratch/kappa4/verify')
from mpmath import mp
import numpy as np, scipy.special as sc
from scipy.stats import kappa4
import vref
mp.dps = 80
warnings.simplefilter('ignore')

print('== F1 mechanism: which operation overflows, and the real regime boundary ==')
for x,h,k in [(-709.0,-1e6,0.0),(-700.0,-1e6,0.0),(-690.0,-1e6,0.0),(-709.9,-1.0,0.0),(-709.0,-1.0,0.0)]:
    y = np.exp(-x) if k==0 else (1.0-k*x)**(1.0/k)
    print(f'  x={x} h={h}: y={y!r}  -h*y={(-h*y)!r}  log1p={sc.log1p(-h*y)!r}  scipy cdf={kappa4.cdf(x,h,k)!r} ref={mp.nstr(vref.cdf(x,h,k),8)}')
print('  -> boundary is log|h| + ly > 709.78, not ly > 709.78 alone')

print('== F7/F8 decomposition at (h,k)=(0.5,1e-8), x=1 ==')
x,h,k = 1.0,0.5,1e-8
print('  1.0/k =', repr(1.0/k), ' exact 1/fl(k) =', mp.nstr(1/mp.mpf(k),20))
one_mkx = 1.0-k*x
print('  fl(1-k*x) =', repr(one_mkx), ' exact =', mp.nstr(1-mp.mpf(k)*x,20),
      ' rel err =', mp.nstr(mp.mpf(one_mkx)/(1-mp.mpf(k)*x)-1,3))
y_d = one_mkx**(1.0/k)
y_e = mp.e**(mp.log1p(-mp.mpf(k)*x)/mp.mpf(k))
print('  y double =', repr(y_d), ' y exact =', mp.nstr(y_e,20), ' rel =', mp.nstr(mp.mpf(y_d)/y_e-1,3))
# how much of that comes from the rounded exponent alone?
y_d2 = float(mp.e**(mp.mpf(1.0/k)*mp.log(mp.mpf(one_mkx))))   # exact power, rounded base+exponent
y_d3 = float(mp.e**((1/mp.mpf(k))*mp.log(1-mp.mpf(k)*x)))     # exact everything
print('  exact power of rounded base/exponent:', repr(y_d2), ' rel to exact:', mp.nstr(mp.mpf(y_d2)/y_e-1,3))
cdf_with_exact_y = np.exp((1.0/h)*sc.log1p(-h*float(y_e)))
print('  cdf(scipy)=', repr(kappa4.cdf(x,h,k)), ' cdf with exact y =', repr(cdf_with_exact_y),
      ' ref =', mp.nstr(vref.cdf(x,h,k),20))

print('== F4/F3 mechanism: stepwise _ppf.f0 ==')
for q,h,k in [(0.9999999999999999,0.5,1e-08),(0.01,0.5,1e-8),(0.1,-1000.0,0.1),(1e-300,1.0,-0.5),(1e-8,1.0,-0.5),(0.9999999999999997,-0.1,0.1)]:
    q = np.float64(q); h = np.float64(h); k = np.float64(k)
    qh = q**h
    num = 1.0-qh
    yv = num/h
    p = yv**k
    res = 1.0/k*(1.0-p)
    print(f'  q={q!r} h={h} k={k}: q**h={qh!r} 1-q**h={num!r} y={yv!r} y**k={p!r} -> {res!r}  scipy={kappa4.ppf(q,h,k)!r} ref={mp.nstr(vref.ppf(q,h,k),17)}')
    print(f'      exact: q**h={mp.nstr(mp.mpf(q)**h,20)} 1-q**h={mp.nstr(-mp.expm1(h*mp.log(mp.mpf(q))),10)} log y={mp.nstr(vref._ly_of_q(q,h),10)}')

print('== F2 mechanism: the two logs ==')
q,h = 0.36,-1.0
print('  q**h =', repr(q**h), ' log1p(-(q**h)) =', repr(sc.log1p(-(q**h))), ' np.log(h) =', repr(np.log(h)),
      ' (1-q**h)/h =', repr((1.0-q**h)/h), ' -log of that =', repr(-np.log((1.0-q**h)/h)))

print('== F6 mechanism: 1-h*y near the endpoint ==')
for x,h,k in [(1e-10,1.0,-0.5),(1e-8,1.0,-0.5),(1e-300,1.0,-0.5),(-2.3025850929940455,0.1,0.0)]:
    y = np.exp(-x) if k==0 else (1.0-k*x)**(1.0/k)
    print(f'  x={x!r} h={h} k={k}: h*y={h*y!r} 1-h*y(double)={1.0-h*y!r} exact 1-h*y={mp.nstr(-mp.expm1(mp.log(mp.mpf(h))+vref._ly(x,k)),10)} log1p(-h*y)={sc.log1p(-h*y)!r}')
