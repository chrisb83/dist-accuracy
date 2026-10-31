# F1: k in (1,2). Nested mp.quad (gk_ref) at reduced precision (20 and 30 digits);
# SciPy returns nan, so a few correct digits suffice to classify.
import sys, warnings, time; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import mpmath as mp, numpy as np
from gk_ref import pdf
from scipy import stats
for x,k,df in [(1.0,1.5,10),(3.0,1.9,10),(1.0,1.5,1e6)]:
    v=[]
    for d in (20,30):
        mp.mp.dps=d; t=time.time(); v.append(pdf(x,k,df)); 
    mp.mp.dps=30
    with np.errstate(all='ignore'):
        sv=float(stats.studentized_range.pdf(x,k,df))
    print(f"pdf({x!r}, {k}, {df!r}) scipy={sv!r} ref={mp.nstr(v[1],17)} 20v30={mp.nstr(abs(v[0]-v[1])/abs(v[1]),3)} t={time.time()-t:.0f}s", flush=True)
