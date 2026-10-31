import numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
d=stats.nct
def tryit(f,*a):
    try: return f"{f(*a):.6e}"
    except Exception as e: return "EXC"
for df in [100,200,300,340,350,1000,1e4,1e6,1e10,1e15,1e16]:
    row=[]
    for x in [-10,-6,-5,-4,5,8,10,15]:
        row.append(f"{x}:{tryit(d.pdf,x,df,2.0)}/{tryit(d.cdf,x,df,2.0)}")
    print(df," ".join(row))
