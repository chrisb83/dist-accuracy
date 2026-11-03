import pickle, glob, warnings, sys, math
import numpy as np
from scipy.stats import genhyperbolic as G
warnings.simplefilter("ignore")
d = {}
for f in glob.glob(".cache/dist-accuracy/genhyperbolic_*_100.pkl"):
    d.update(pickle.load(open(f, "rb")))
meth = sys.argv[1:] or ["cdf", "sf"]
rows = []
for (m, x, sh), r in d.items():
    if m not in meth or r is None: continue
    rf = float(r)
    if not (2.2e-308 <= abs(rf) <= 1.8e308): continue
    s = float(getattr(G, m)(x, *sh))
    u = abs(s - rf) / abs(rf) / 2.22e-16 if np.isfinite(s) else math.inf
    rows.append((u, m, sh, x, s, rf))
rows.sort(key=lambda t: (t[1], t[2], -t[0]))
from collections import defaultdict
g = defaultdict(list)
for t in rows: g[(t[1], t[2])].append(t)
for k, v in g.items():
    us = sorted(t[0] for t in v)
    print(k, "n", len(v), "median ulp %.3g" % us[len(us)//2], "max %.3g" % us[-1])
    for t in v[:8]: print("   %.3g ulp  x=%r scipy=%r ref=%r" % (t[0], t[3], t[4], t[5]))
