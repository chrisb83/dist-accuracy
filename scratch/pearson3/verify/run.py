import sys, time, warnings
import numpy as np, mpmath as mp
from scipy import stats
import vref
warnings.simplefilter("ignore")
CASES = [
 ("F1","ppf",1e-20,-2),("F1","ppf",1e-300,-2),("F1","ppf",3.4807005884285624e-14,-2),("F1","ppf",1e-12,-0.1),
 ("F2","logpdf",40.0,1.59e-5),("F2","logpdf",100.0,1.59e-5),
 ("F3","ppf",0.5,1.59e-5),("F3","cdf",-30.0,-1.59e-5),("F3","cdf",-3.0,1.59e-5),("F3","pdf",3.0,1.59e-5),
 ("F4","cdf",-5.0,1.6e-5),("F4","cdf",-8.0,1.6e-5),("F4","sf",8.0,-1.6e-5),("F4","cdf",-6.0,1e-3),("F4","ppf",1e-10,1.6e-5),
 ("F5","sf",-0.2,10.0),("F5","cdf",0.2,-10.0),
 ("F6","pdf",0.0,1.6e-5),("F6","logpdf",0.0,1.6e-5),("F6","pdf",0.0,1e-3),("F6","pdf",-20.0,0.01),
 ("F7","ppf",0.5,1.6e-5),("F7","ppf",0.5,0.01),
 ("F8","cdf",-3.3,1.7e-5),
 ("F9","cdf",-37.0,-0.1),("F9","cdf",-9.0,0.1),
]
sel = sys.argv[1:] 
def ref(m, a, s):
    if m == "ppf":
        x0 = stats.norm.ppf(a) if abs(s) < 1 else stats.pearson3.ppf(a, s)
        if not np.isfinite(x0): x0 = stats.norm.ppf(a) if a > 1e-200 else -600.0
        return vref.ppf(a, s, x0)
    if m == "cdf": return vref.cdf_edge(a, s)
    if m == "sf": return vref.sf_edge(a, s)
    return getattr(vref, m)(a, s)
for fid, m, a, s in CASES:
    if sel and fid not in sel: continue
    t0 = time.time()
    sp = getattr(stats.pearson3, m)(a, s)
    mp.mp.dps = 100; r1 = ref(m, a, s)
    mp.mp.dps = 200; r2 = ref(m, a, s)
    mp.mp.dps = 50
    agree = mp.nstr(abs(r1 - r2) / abs(r2), 3) if r2 != 0 else "exact"
    ulp = float(abs(mp.mpf(sp) - r2) / abs(r2) / 2.220446049250313e-16) if np.isfinite(sp) else float("inf")
    print(f"{fid} {m}({a!r}, {s!r}) scipy={sp!r} ref={mp.nstr(r2, 17)} rel100v200={agree} ulp={ulp:.3g} t={time.time()-t0:.1f}s", flush=True)
