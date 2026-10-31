import sys, time; sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/refs")
import mpmath as mp, ncf_ref as R
from scipy import stats
mp.mp.dps = 100
cnt=[0]
of=R.cdf; osf=R.sf
def c(*a):
    cnt[0]+=1; return of(*a)
def s_(*a):
    cnt[0]+=1; return osf(*a)
R.cdf=c; R.sf=s_
for s in R.PARAMS:
    for q in [1e-100, 0.3, 0.9, 1-1e-10]:
        cnt[0]=0
        t=time.time(); v=R.ppf(q,*s); 
        print(s, q, float(v), stats.ncf.ppf(q,*s), f"{time.time()-t:.2f}s calls={cnt[0]}", flush=True)
