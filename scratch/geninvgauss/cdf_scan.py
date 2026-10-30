import sys, warnings
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
warnings.simplefilter('ignore')
import numpy as np, mpmath as mp
mp.mp.dps = 100
import geninvgauss_ref as r
from scipy.stats import geninvgauss as g
for s in [(2.3, 1.5), (-0.5, 1.0), (0.0, 1e-3), (0.5, 1e4)]:
    q = [1e-12, 1e-8, 1e-5, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    xs = list(g.ppf(q, *s)) + list(r.x_points(*s))
    rows = []
    for x in xs:
        if not (np.isfinite(x) and x > 0):
            continue
        ref = r.cdf(mp.mpf(x), *map(mp.mpf, s))
        v = g.cdf(x, *s)
        ulp = float(abs(v - ref) / ref / 2.220446049250313e-16)
        rows.append((ulp, x, v, float(ref)))
    rows.sort(reverse=True)
    for row in rows[:6]:
        print(s, 'ulp %.3g x %r scipy %r ref %r' % row)
    sys.stdout.flush()
