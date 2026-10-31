import sys; sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/refs")
import numpy as np, mpmath as mp, ncf_ref as R
from scipy import stats
mp.mp.dps = 100
qs=[1e-300,1e-200,1e-100,1e-50,1e-20,1e-10,1e-5,0.1,0.5,0.9,1-1e-10]
for s in R.PARAMS:
    for q in qs:
        try: v=float(stats.ncf.isf(q,*s))
        except Exception as e: v=type(e).__name__
        r=R.isf(q,*s)
        if isinstance(v,float) and np.isfinite(v) and v!=0: u=f"{float(abs(v-r)/r)/2.22e-16:.3g}"
        else: u="-"
        print(s,q,v,mp.nstr(r,17),u,flush=True)
