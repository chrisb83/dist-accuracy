import sys; sys.path.insert(0, "refs")
import mpmath as mp, numpy as np
from scipy import stats
import chi2_ref as R
mp.mp.dps = 400
for name, x, k in [("logsf", 1600.0, 55.0), ("logsf", 4000.0, 2.0), ("logcdf", 1e-300, 55.0), ("logcdf", 200.0, 2e3)]:
    with np.errstate(all="ignore"):
        v = getattr(stats.chi2, name)(x, k)
    r = R.REFS[name](mp.mpf(x), mp.mpf(k))
    print(name, x, k, repr(float(v)), mp.nstr(r, 17))
