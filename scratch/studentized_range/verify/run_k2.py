import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import mpmath as mp, numpy as np
from k2_ref import cdfA, pdfA, cdfB, pdfB
from scipy import stats
pts = [('pdf',1e5,0.1),('pdf',1e4,1),('cdf',1e6,0.1),('cdf',1e5,1),
       ('cdf',1e-18,10),('cdf',1e-8,10),('cdf',1e-6,10),
       ('pdf',1e10,10),('pdf',100,10),('pdf',10,10),
       ('pdf',30,1e5),('pdf',1,1e5),('cdf',1,1e5),
       ('pdf',9.5,99999),('cdf',1e-10,99999),
       ('cdf',1,99999),('pdf',1,99999)]
for m,x,df in pts:
    A = cdfA if m=='cdf' else pdfA
    B = cdfB if m=='cdf' else pdfB
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append((A(x,df), B(x,df)))
    mp.mp.dps = 100
    a1,b1 = vals[0]; a2,b2 = vals[1]
    agreeA = mp.nstr(abs(a1-a2)/abs(a2), 3); agreeAB = mp.nstr(abs(a2-b2)/abs(b2), 3)
    with np.errstate(all='ignore'):
        sv = float(getattr(stats.studentized_range, m)(x, 2, df))
    ref = b2
    ulp = float(abs(mp.mpf(sv)-ref)/(abs(ref)*mp.mpf(2)**-52))
    print(f"{m}({x!r}, 2, {df!r}) scipy={sv!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} A100v200={agreeA} AvsB={agreeAB}", flush=True)
