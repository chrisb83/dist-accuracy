import numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
for df,nc in [(10,40),(30,20),(3,38),(100,30),(5,50),(10,37),(10,38.5),(1000,30)]:
    xs=np.array([1,10,30,50,100,1e3,1e4,1e5,1e10,1e20,1e50,1e77,1e80,1e100,1e150])
    out=[]
    for x in xs:
        try: out.append(f"{stats.nct.pdf(x,df,nc):.3e}")
        except Exception as e: out.append("EXC")
    print(df,nc," ".join(out))
