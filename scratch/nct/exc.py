import numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
d=stats.nct
for df,nc in [(1,1),(0.5,5),(30,20),(1e6,2),(1e5,2),(1e4,2),(14,0.24)]:
    for x in [-1e300,-1e154,-1.35e154,-1.3e154,-1e5,-10,-3,10,1e5,1e154,1.35e154,1e200]:
        r=[]
        for m in ["pdf","cdf","sf"]:
            try: r.append(f"{m}={getattr(d,m)(x,df,nc):.4e}")
            except Exception as e: r.append(f"{m}=EXC({str(e)[17:50]})")
        print(df,nc,x," ".join(r))
for df,nc in [(1e6,2),(1e5,2),(1,1),(0.5,5)]:
    for q in [1e-300,1e-20,1e-5,0.5,0.99999]:
        r=[]
        for m in ["ppf","isf"]:
            try: r.append(f"{m}={getattr(d,m)(q,df,nc):.6e}")
            except Exception as e: r.append(f"{m}=EXC({str(e)[17:50]})")
        print(df,nc,q," ".join(r))
