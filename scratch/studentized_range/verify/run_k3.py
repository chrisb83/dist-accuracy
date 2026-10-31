import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import mpmath as mp, numpy as np
from k3_ref import pdf3, cdf3, r3
from gk_ref import rk
from scipy import stats
mp.mp.dps = 50
print('closed-form r3 vs nested quad at w=0.7:', mp.nstr(r3(mp.mpf('0.7'))-rk(mp.mpf('0.7'),3), 3), flush=True)
for m,x in [('pdf',1.0),('cdf',1.0),('pdf',1e-6),('pdf',1e-9),('cdf',1e-4),('cdf',1e-8)]:
    f = pdf3 if m=='pdf' else cdf3
    v=[]
    for d in (100,200):
        mp.mp.dps=d; v.append(f(x,10))
    mp.mp.dps=100
    with np.errstate(all='ignore'):
        sv=float(getattr(stats.studentized_range,m)(x,3,10))
    ref=v[1]; ulp=float(abs(mp.mpf(sv)-ref)/(abs(ref)*mp.mpf(2)**-52))
    print(f"{m}({x!r}, 3, 10) scipy={sv!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} 100v200={mp.nstr(abs(v[0]-v[1])/abs(ref),3)}", flush=True)
