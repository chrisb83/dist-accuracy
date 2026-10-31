"""Scan cached ppf/isf references; ulp vs SciPy; analytic cond = q/(x f(x))."""
import pickle, sys, warnings
import numpy as np, mpmath as mp
sys.path.insert(0, 'refs')
import ncx2_ref as R
from scipy import stats
mp.mp.dps = 50
d = pickle.load(open(sys.argv[1], 'rb'))
rows = []
for (m, q, s), v in d.items():
    if m not in ('ppf', 'isf') or v is None:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            got = float(getattr(stats.ncx2, m)(q, *s))
        except Exception as e:
            got = type(e).__name__
    ref = float(v)
    if not (2.2e-308 < ref < 1.8e308):
        continue
    if isinstance(got, str) or not np.isfinite(got) or got <= 0:
        ulp = np.inf
    else:
        ulp = abs(got - ref) / (ref * 2.22e-16)
    qq = q
    try:
        cond = float(mp.mpf(qq) / (v * R.pdf(v, *s)))
    except Exception:
        cond = float('nan')
    rows.append((m, s, q, got, ref, ulp, cond))
rows.sort(key=lambda r: (r[0], r[1], r[2]))
for r in rows:
    m, s, q, got, ref, ulp, cond = r
    if ulp > 8 + cond:
        print(f"{m} {s} q={q:.6g} scipy={got!r} ref={ref!r} ulp={ulp:.3g} cond={cond:.3g}")
print('max ulp-cond excess per method/set:')
from collections import defaultdict
mx = defaultdict(float)
for m, s, q, got, ref, ulp, cond in rows:
    mx[(m, s)] = max(mx[(m, s)], ulp)
for k, v in sorted(mx.items()): print(k, f"{v:.3g}")
