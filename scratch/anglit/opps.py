import sys; sys.path.insert(0, "refs")
import warnings, numpy as np, mpmath as mp
from scipy import stats
import anglit_ref as R
mp.mp.dps = 100
d = stats.anglit
for name, f, arg in [("logpdf", d.logpdf, 1e-9), ("logcdf", d.logcdf, np.pi/4 - 1e-9),
                     ("logsf", d.logsf, -np.pi/4 + 1e-9), ("isf", d.isf, 1e-18), ("isf", d.isf, 1e-10)]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = f(arg)
    ref = R.REFS[name](mp.mpf(float(arg)))
    rel = abs((mp.mpf(float(v)) - ref) / ref)
    print(name, repr(arg), repr(float(v)), mp.nstr(ref, 17), "ulp=%.2g" % float(rel / 2**-52), [str(x.message) for x in w])
# candidate formulas
x = 1e-9; print("log1p(-2 sin^2 x):", np.log1p(-2*np.sin(x)**2))
x = np.pi/4 - 1e-9; print("log1p(-sf):", np.log1p(-d.sf(x)))
q = 1e-18; print("pi/4 - arcsin(sqrt q):", np.pi/4 - np.arcsin(np.sqrt(q)))
