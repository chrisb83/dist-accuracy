"""Candidate ppf near the median: sign(q-1/2)*sqrt(betaincinv(1/2, c/2, |2q-1|)) (X**2 ~ Beta(1/2, c/2))."""
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, numpy as np, scipy.special as sc
from scipy import stats
import rdist_ref as r
mp.mp.dps = 120
def u(v, ref): return float(abs((mp.mpf(v) - ref) / ref)) / 2.22e-16 if np.isfinite(v) and v != 0 else float('inf')
for c in [0.001, 0.5, 1.6, 3.0, 10.0, 1e3, 1e6]:
    out = []
    for q in [0.4999999999999999, 0.5000000000000001, 0.49999999999999, 0.4999, 0.4, 0.6, 0.3]:
        ref = r.ppf(mp.mpf(q), c)
        t = abs(2 * q - 1)
        cand = np.sign(q - 0.5) * np.sqrt(sc.betaincinv(0.5, c / 2, t))
        out.append(f"{q}: scipy {u(stats.rdist.ppf(q, c), ref):.3g} cand {u(cand, ref):.3g}")
    print(f"c={c:g}: " + "; ".join(out))
