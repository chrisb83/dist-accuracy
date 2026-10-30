import pickle, glob, os, numpy as np
from scipy import stats
f = max(glob.glob('.cache/dist-accuracy/invgamma_*_400.pkl'), key=os.path.getmtime)
c = pickle.load(open(f, 'rb'))
rows = []
for (m, arg, sh), v in c.items():
    if m not in ('ppf', 'isf') or v is None: continue
    r = float(v)
    if not (2.2e-308 < abs(r) < 1.8e308): continue
    with np.errstate(all='ignore'):
        s = float(getattr(stats.invgamma, m)(arg, *sh))
    u = abs(s - r) / (abs(r) * 2.22e-16) if np.isfinite(s) else np.inf
    rows.append((u, m, arg, sh, s, r))
rows.sort(key=lambda t: -t[0])
for t in rows[:40]: print('%9.3g %s %r %s %r %r' % t)
