import time
from mpmath import mp
mp.dps = 60
for a, x in [(1e7, 1e7-5*mp.sqrt(mp.mpf(1e7))), (1e8, 99954000.0)]:
    t0=time.time()
    try:
        v = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True, maxterms=10**7)
        print(a, mp.nstr(v,25), "%.1fs"%(time.time()-t0))
    except Exception as e:
        print(a, "ERR", type(e).__name__, str(e)[:100], "%.1fs"%(time.time()-t0))
