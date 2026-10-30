# Independent reference for gibrat logpdf: log of the docstring pdf,
# f(x) = exp(-log(x)**2/2) / (x*sqrt(2*pi)), evaluated as mp.log(pdf) (route differs
# from the analyst's summed-log form). Two precisions, must agree to 30 digits.
import warnings
import mpmath as mp
import numpy as np
from scipy import stats

def ref(x, dps):
    with mp.workdps(dps):
        x = mp.mpf(x)
        pdf = mp.npdf(mp.log(x)) / x   # standard normal pdf of log x, divided by x
        return mp.log(pdf)

pts = [8e307, 1.7e308, 5e-324, 1e-322, 1e-320, 1e-315]
for x in pts:
    r100, r200 = ref(x, 100), ref(x, 200)
    with mp.workdps(200):
        agree = abs(r100 - r200) / abs(r200) < mp.mpf(10)**-30
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        s = stats.gibrat.logpdf(x)
    r = float(r200)
    ulp = abs(s - r) / (abs(r) * 2.220446049250313e-16) if np.isfinite(s) else np.inf
    print(f"x={x!r}: scipy={s!r} ref={r!r} ({mp.nstr(r200, 20)}) agree30={agree} ulp={ulp:.3g}")
    # product check for mechanism
    with np.errstate(all="ignore"):
        prod = x * np.sqrt(2*np.pi)
    with mp.workdps(50):
        exact = mp.mpf(x)*mp.sqrt(2*mp.pi)
        print("   product rel err:", mp.nstr(abs(mp.mpf(prod)-exact)/exact, 3) if np.isfinite(prod) else "inf")
