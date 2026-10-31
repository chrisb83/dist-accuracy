import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import mpmath as mp, numpy as np
from k3_owen import pdfO, cdfO, cdfP
from k3_ref import pdf3
from scipy import stats
mp.mp.dps=60
print('pdfO vs pdf3 (r3 route) at x=1:', mp.nstr(pdfO(1,10)-pdf3(1,10),3), ' x=1e-6:', mp.nstr((pdfO(1e-6,10)-pdf3(1e-6,10))/pdfO(1e-6,10),3))
print('cdfO vs int pdf at x=1:', mp.nstr(cdfO(1,10)-cdfP(1,10),3))
for m,x in [('pdf',1e-6),('pdf',1e-9),('cdf',1e-4),('cdf',1e-8),('cdf',1.0)]:
    v=[]
    for d in (100,200):
        mp.mp.dps=d
        v.append(pdfO(x,10) if m=='pdf' else cdfP(x,10))
    mp.mp.dps=100
    chk = '' if m=='pdf' else ' cdfO-diff=' + mp.nstr(abs(cdfO(x,10)-v[0])/v[0],3)
    with np.errstate(all='ignore'):
        sv=float(getattr(stats.studentized_range,m)(x,3,10))
    ref=v[1]; ulp=float(abs(mp.mpf(sv)-ref)/(abs(ref)*mp.mpf(2)**-52))
    print(f"{m}({x!r}, 3, 10) scipy={sv!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} 100v200={mp.nstr(abs(v[0]-v[1])/abs(ref),3)}{chk}", flush=True)
