import sys, time, mpmath as mp
sys.path.insert(0, "refs"); import nct_ref as R
mp.mp.dps = 100
for k,c in [(10,30),(30,20)]:
    for q in [1e-300,1e-10,0.3,0.9,1-1e-10]:
        t=time.time(); a=R.ppf(q,k,c); b=R.isf(q,k,c); print(k,c,q,mp.nstr(a,17),mp.nstr(b,17),f"{time.time()-t:.1f}s")
