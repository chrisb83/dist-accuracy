import sys, mpmath as mp, warnings
sys.path.insert(0, "refs"); import nct_ref as R
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 50
def u(s, r): r=float(r); return abs(s-r)/abs(r)/2.22e-16 if r else float('nan')
for k in [100, 300, 1e4, 1e8, 1e12, 1e14, 1e15, 4e15, 5e15, 1e16]:
    for x in [-4.0, 0.5, 2.0, 5.0]:
        rp = R.pdf(x,k,2.0); rc = R.cdf(x,k,2.0)
        try: sp = stats.nct.pdf(x,k,2.0)
        except Exception: sp = float('nan')
        sc_ = stats.nct.cdf(x,k,2.0)
        print(f"k={k:g} x={x} pdf {sp:.10e} ref {mp.nstr(rp,11)} ulp {u(sp,rp):.2e} | cdf {sc_:.10e} ref {mp.nstr(rc,11)} ulp {u(sc_,rc):.2e}")
