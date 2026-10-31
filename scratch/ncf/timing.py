import sys, time; sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/refs")
import mpmath as mp, ncf_ref as R
from scipy import stats
mp.mp.dps = 100
for s in R.PARAMS:
    for x in [1e-300, 0.5, 1.5, 1e10, 1e300]:
        out=[]
        for meth in ["pdf","cdf","sf"]:
            t=time.time(); v=R.REFS[meth](x,*s); dt=time.time()-t
            sv=getattr(stats.ncf, meth)(x,*s)
            out.append(f"{meth} {float(v):.6e} sp={sv:.6e} {dt:.3f}s")
        print(s, x, " | ".join(out))
    t=time.time(); q=R.ppf(0.3,*s); print(" ppf(0.3)", float(q), stats.ncf.ppf(0.3,*s), f"{time.time()-t:.2f}s")
