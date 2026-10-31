import numpy as np, warnings
from scipy import stats, special as sc
import scipy.special._ufuncs as scu
warnings.simplefilter("ignore")
d=stats.nct
for df,nc in [(14,0.24),(1,1),(5,-3),(0.5,5),(30,20),(3,38),(10,40),(1e6,2),(10,0)]:
    print("==",df,nc)
    for x in [-1e300,-1e200,-1e154,-1e100,-1e20,-1e5,-10,-1,0,1,10,1e5,1e20,1e100,1e154,1e200,1e300]:
        try:
            print(f"{x:9.0e} pdf {d.pdf(x,df,nc):.6e} cdf {d.cdf(x,df,nc):.6e} sf {d.sf(x,df,nc):.6e}")
        except Exception as e: print(x, repr(e))
    for q in [1e-300,1e-100,1e-10,0.5,1-1e-10]:
        try: print(f"q={q:.0e} ppf {d.ppf(q,df,nc):.6e} isf {d.isf(q,df,nc):.6e}")
        except Exception as e: print(q, repr(e))
