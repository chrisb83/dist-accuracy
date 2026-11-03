"""One demonstration point per inherited method (generic default vs reference)."""
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, numpy as np, scipy.special as sc
from scipy import stats
import rdist_ref as r
mp.mp.dps = 120
def u(v, ref): return float(abs((mp.mpf(v) - ref) / ref)) / 2.22e-16 if np.isfinite(v) else float('inf')
for name, call, refv, alt in [
    ("isf(1e-20, 1e6)", lambda: stats.rdist.isf(1e-20, 1e6), r.isf(mp.mpf(1e-20), 1e6), lambda: -stats.rdist.ppf(1e-20, 1e6)),
    ("isf(1e-10, 3)", lambda: stats.rdist.isf(1e-10, 3), r.isf(mp.mpf(1e-10), 3), lambda: -stats.rdist.ppf(1e-10, 3)),
    ("logcdf(-0.05, 1e6)", lambda: stats.rdist.logcdf(-0.05, 1e6), r.logcdf(mp.mpf(-0.05), 1e6), None),
    ("logsf(0.05, 1e6)", lambda: stats.rdist.logsf(0.05, 1e6), r.logsf(mp.mpf(0.05), 1e6), None),
    ("logcdf(-0.9, 1000)", lambda: stats.rdist.logcdf(-0.9, 1000), r.logcdf(mp.mpf(-0.9), 1000), None),
]:
    v = call()
    line = f"{name}: scipy {v!r} ref {mp.nstr(refv, 17)} ({u(v, refv):.3g} ulp)"
    if alt is not None:
        a = alt(); line += f"; -ppf(q): {a!r} ({u(a, refv):.3g} ulp)"
    print(line)
