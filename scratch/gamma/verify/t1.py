import time
from mpmath import mp
for dps in (60,):
    mp.dps = dps
    for a, x in [(1e6, 995000.0), (1e7, 1e7-5*3162.27766), (1e8, 99954000.0)]:
        t0=time.time()
        try:
            v = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True)
            print(dps, a, x, mp.nstr(v, 25), "%.1fs"%(time.time()-t0))
        except Exception as e:
            print(dps, a, x, "ERR", type(e).__name__, str(e)[:80], "%.1fs"%(time.time()-t0))
