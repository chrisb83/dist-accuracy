import sys, time, numpy as np, mpmath as mp
sys.path.insert(0,'refs'); sys.path.insert(0,'.claude/skills/dist-accuracy-review/scripts')
import kappa4_ref as r
from accuracy import default_x
from scipy.stats import kappa4
mp.mp.dps=400
shapes=(0.1,0.0)
P=[mp.mpf(s) for s in shapes]
bulk=[float(r.ppf(mp.mpf(q),*P)) for q in (0.1,0.5,0.9)]
print('bulk',bulk, flush=True)
grid=default_x(kappa4, shapes)
print('grid ends', grid[0], grid[1], grid[-2], grid[-1], flush=True)
for m in r.REFS:
    for x in bulk+[grid[1],grid[-2]]:
        t0=time.time()
        try:
            lo=r.REFS[m](mp.mpf(x),*P)
        except Exception as e:
            lo=repr(e)
        print(m, x, mp.nstr(lo,8) if not isinstance(lo,str) else lo, '%.2f'%(time.time()-t0), flush=True)
