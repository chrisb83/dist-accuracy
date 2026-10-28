import sys, warnings
sys.path.insert(0,'C:/Users/chbau/Documents/research/dist-accuracy/scratch/kappa4/verify')
from mpmath import mp
import numpy as np
from scipy.stats import kappa4
import vref
mp.dps = 120
EPS = 2.220446049250313e-16
warnings.simplefilter('ignore')

print('--- F2: ppf for k=0, h<0 over a q grid ---')
qs = np.array([1e-300,1e-100,1e-20,1e-8,1e-3,0.01,0.1,0.25,0.36,0.5,0.75,0.9,0.99,
               1-1e-3,1-1e-8,1-1e-15,0.9999999999999997])
for h in (-1.0, -0.5, -1e6, -1e-3):
    sv = kappa4.ppf(qs, h, 0.0)
    nn = int(np.sum(np.isnan(sv)))
    print(f'  h={h}: nan for {nn}/{len(qs)} q; e.g. q=0.5 -> scipy {sv[9]!r}, ref {mp.nstr(vref.ppf(0.5,h,0.0),17)}')
print('  logistic check: kappa4.ppf(0.36,-1,0) =', kappa4.ppf(0.36,-1.0,0.0),
      ' scipy.stats.logistic.ppf(0.36) =', __import__('scipy.stats',fromlist=['logistic']).logistic.ppf(0.36),
      ' ref =', mp.nstr(vref.ppf(0.36,-1.0,0.0),17))
print('  kappa4.cdf(ref_x,-1,0) =', kappa4.cdf(float(vref.ppf(0.36,-1.0,0.0)),-1.0,0.0))

print('--- conditioning (cond_x = |x f/F| for cdf; |x f/(F log F)| for logcdf; cond_q = |q/(x f)| for ppf) ---')
def cond_cdf(x,h,k):
    return abs(mp.mpf(x)*vref.pdf(x,h,k)/vref.cdf(x,h,k))
def cond_logcdf(x,h,k):
    return abs(mp.mpf(x)*vref.pdf(x,h,k)/(vref.cdf(x,h,k)*vref.logcdf(x,h,k)))
def cond_ppf(q,h,k):
    x = vref.ppf(q,h,k); return abs(mp.mpf(q)/(x*vref.pdf(x,h,k)))
for lbl,f,a in [('cdf(-709,-1e6,0)',cond_cdf,(-709.0,-1e6,0.0)),
                ('logcdf(-709,-1e6,0)',cond_logcdf,(-709.0,-1e6,0.0)),
                ('cdf(-6.70e31,-1000,0.1)',cond_cdf,(-6.702276267921237e+31,-1000.0,0.1)),
                ('cdf(-99.946,-1e6,-0.01)',cond_cdf,(-99.94613906403504,-1e6,-0.01)),
                ('pdf cond n/a',None,None),
                ('cdf(1e-300,1,-0.5)',cond_cdf,(1e-300,1.0,-0.5)),
                ('cdf(1e-10,1,-0.5)',cond_cdf,(1e-10,1.0,-0.5)),
                ('cdf(1,0.5,1e-8)',cond_cdf,(1.0,0.5,1e-8)),
                ('logcdf(-0.5,0.5,1e-8)',cond_logcdf,(-0.5,0.5,1e-8)),
                ('ppf(0.36,-1,0)',cond_ppf,(0.36,-1.0,0.0)),
                ('ppf(0.3,1e-8,0.5)',cond_ppf,(0.3,1e-8,0.5)),
                ('ppf(0.9999999999999999,0.5,1e-8)',cond_ppf,(0.9999999999999999,0.5,1e-8)),
                ('ppf(0.1,-1000,0.1)',cond_ppf,(0.1,-1000.0,0.1)),
                ('ppf(1e-8,1,-0.5)',cond_ppf,(1e-8,1.0,-0.5))]:
    if f is None: continue
    print(f'  cond {lbl} = {mp.nstr(f(*a),3)}')

print('--- _get_support lower endpoint ---')
for h,k in [(0.5,1e-8),(2.0,0.5),(1.0,-0.5),(0.1,0.0),(0.5,1e-4),(3.0,1e-10)]:
    a_sc, b_sc = kappa4._get_support(np.array([h]), np.array([k]))
    a_ref = vref.support_lower(h,k)
    rel = abs(mp.mpf(float(a_sc[0]))/a_ref - 1) if a_ref != 0 else mp.inf
    print(f'  h={h} k={k}: scipy a={a_sc[0]!r} ref={mp.nstr(a_ref,20)} rel={mp.nstr(rel,3)} ulp={mp.nstr(rel/EPS,3)}')
print('  pdf/cdf just above the true endpoint for (0.5,1e-8):')
a_sc = float(kappa4._get_support(np.array([0.5]), np.array([1e-8]))[0][0])
a_true = float(vref.support_lower(0.5,1e-8))
print('   scipy a =', repr(a_sc), ' true a =', repr(a_true), ' a_sc < a_true ?', a_sc < a_true)
for x in [a_sc, np.nextafter(a_sc,0), -0.6931471740943004, -0.69314718, a_true, np.nextafter(a_true,1)]:
    try: r = mp.nstr(vref.pdf(x,0.5,1e-8),17)
    except Exception as e: r = f'outside support'
    print(f'   x={x!r}: scipy pdf={kappa4.pdf(x,0.5,1e-8)!r} cdf={kappa4.cdf(x,0.5,1e-8)!r} refpdf={r}')
