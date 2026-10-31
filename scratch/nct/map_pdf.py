import sys, mpmath as mp, warnings
sys.path.insert(0, "refs"); import nct_ref as R
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 60
def u(s, r): r=float(r); return abs(s-r)/abs(r)/2.22e-16 if r else float('nan')
for k,d in [(10,3),(10,8),(10,20),(30,20),(10,30)]:
    for t in [-0.5,-3,-10,-100,-1e4]:
        r = R.pdf(t,k,d)
        try: s = stats.nct.pdf(t,k,d)
        except Exception as e: s=float('nan')
        print(f"pdf k={k} d={d} t={t} scipy={s:.10e} ref={mp.nstr(r,11)} ulp={u(s,r):.2e}")
for k,d in [(10,-3),(10,-8),(5,-3)]:
    for t in [0.5,3,10,100,1e4]:
        r = R.sf(t,k,d); s = stats.nct.sf(t,k,d)
        print(f"sf k={k} d={d} t={t} scipy={s:.10e} ref={mp.nstr(r,11)} ulp={u(s,r):.2e}")
