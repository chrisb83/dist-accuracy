import numpy as np, scipy.special as sc
from scipy.stats import loggamma as L
np.seterr(all='ignore')
calls = [('logpdf',0.0,1e-310),('cdf',-1.0,1e-310),('ppf',0.99,1e-310),('isf',0.01,1e-310),
 ('cdf',13.810910557964274,1e6),('cdf',13.809510557964273,1e6),('cdf',13.805510557964274,1e6),('sf',13.810910557964274,1e6),
 ('sf',-800.0,1e-14),('sf',-708.3964185322642,1e-10),
 ('pdf',13.845748741024885,1e6),('pdf',13.815210557964274,1e6),('logpdf',13.818510557964274,1e6),
 ('isf',0.9999993371296838,1e6),('ppf',1e-08,1e6)]
for m,x,c in calls: print(m,x,c,repr(getattr(L,m)(x,c)))
print('gammaln(1e-310)',sc.gammaln(1e-310),'gammainc',sc.gammainc(1e-310,np.exp(-1)),sc.gammaincinv(1e-310,0.5),sc.gammainccinv(1e-310,0.5))
print('gammaln(1+1e-14)',sc.gammaln(1+1e-14))
