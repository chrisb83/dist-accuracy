import sys, mpmath as mp, warnings
sys.path.insert(0, "refs"); import nct_ref as R
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 60
def cross(t,k,d):
    x=t*t/(k+t*t); b=k/2; c=0.5+b+d*d/2; return x, 1-(b/c)*(1+d*d/(2*c*c))
for k,d in [(10,1),(10,3),(10,5),(10,8),(30,20),(3,10),(1e3,5)]:
    for t in [-0.1,-1,-3,-10,-100]:
        r = R.cdf(t,k,d); s = stats.nct.cdf(t,k,d)
        x,cr = cross(t,k,d)
        err = abs(s-float(r))/float(r)/2.22e-16 if float(r) else float('nan')
        print(f"k={k} d={d} t={t} x={x:.3f} cross={cr:.3f} ref={mp.nstr(r,8)} scipy={s:.8e} ulp={err:.2e}")
