import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/t/verify')
import mpmath as mp, numpy as np
from scipy import stats
import vref as V
inf = float('inf'); df274 = 2.7433514990818093
cases = [
 ('F1','cdf',-1e300,0.001),('F1','cdf',1.4e154,0.001),('F1','sf',-1.4e154,0.001),
 ('F1','cdf',-1e200,0.1),('F1','cdf',-1.4e154,1.0),('F1','sf',1.351287904186225e154,1.0),
 ('F2','logpdf',1.4e154,df274),('F2','logpdf',-1e300,30.0),('F2','pdf',-1e300,0.001),
 ('F2','pdf',4.273147551863848e152,0.001),
 ('F3','logpdf',1.4e154,inf),
 ('F4','ppf',0.29,0.001),('F4','ppf',1e-20,0.1),('F4','isf',0.29,0.001),
 ('F5','ppf',1e-308,df274),('F5','isf',1e-308,df274),('F5','ppf',5.6234132519034905e-226,df274),
 ('F5','ppf',3.162277660168379e-151,df274),
 ('F6','cdf',-1e-8,1.0),('F6','sf',1e-8,1.0),('F6','cdf',-1e-10,1.0),('F6','cdf',1e-6,1.0),
 ('F7','ppf',0.795,2.9),('F7','isf',0.205,2.9),
 ('F8','ppf',0.42999999999999994,0.001),
]
def ref(m, a, df):
    if m=='logpdf': return V.logpdf(a, df)
    if m=='pdf': return V.pdf(a, df)
    if m=='cdf': return V.cdf_quad(a, df)
    if m=='sf': return 1 - V.cdf_quad(a, df)
    if m=='ppf': return V.ppf(a, df)
    if m=='isf': return -V.ppf(a, df)
for fid, m, a, df in cases:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(ref(m, a, df))
    r1, r2 = vals
    agree = (abs(r1-r2) <= abs(r2)*mp.mpf(10)**-30)
    mp.mp.dps = 200
    extra = ''
    if m in ('ppf','isf'):
        q = mp.mpf(a); x = r2 if m=='ppf' else -r2
        Fq = V.cdf_quad(x, df) if q < 0.5 else V.cdf_quad(x, df)
        target = q
        extra = ' cdfchk_rel=%s' % mp.nstr(abs(Fq-target)/target, 3)
    s = getattr(stats.t, m)(a, df)
    if np.isfinite(s) and r2 != 0:
        ulp = float(abs(mp.mpf(s)-r2)/abs(r2)/mp.mpf(2.220446049250313e-16))
    else:
        ulp = inf
    print(fid, '%s(%r, %r)' % (m, a, df), 'scipy=%r' % float(s), 'ref=%s' % mp.nstr(r2, 17),
          'ulp=%.3g' % ulp, 'agree30=%s' % agree, extra, flush=True)
