import sys, time
sys.path.insert(0, 'refs'); sys.path.insert(0, '.claude/skills/dist-accuracy-review/scripts')
import numpy as np, mpmath as mp
import scipy.stats as st
from accuracy import default_x, default_q
import exponpow_ref as r
mp.mp.dps = 400
b = float(sys.argv[1])
xs = np.unique(np.concatenate([default_x(st.exponpow, (b,)), r.x_points(b)]))
xs = xs[np.isfinite(xs)]
for m in ['cdf', 'pdf', 'logpdf', 'sf']:
    for x in xs:
        print("start", m, x, flush=True); t = time.time(); r.REFS[m](mp.mpf(x), mp.mpf(b)); dt = time.time() - t
        if dt > 0.05: print(m, x, dt, flush=True)
for m in ['ppf', 'isf']:
    for q in default_q():
        t = time.time(); r.REFS[m](mp.mpf(q), mp.mpf(b)); dt = time.time() - t
        if dt > 0.05: print(m, q, dt, flush=True)
print('done')
