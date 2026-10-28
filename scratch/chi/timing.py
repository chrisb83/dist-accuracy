import sys, time; sys.path.insert(0, 'refs')
import mpmath as mp; mp.mp.dps = 400
import chi_ref as R
for df in (78., 1e-3, 2e6):
    for m, a in (("cdf", 1e-200), ("cdf", (df)**.5*0.99), ("sf", (df)**.5*1.01), ("sf", 30.), ("ppf", 1e-100), ("isf", 1e-100), ("ppf", 0.5)):
        t = time.time(); v = R.REFS[m](mp.mpf(a), df); print(df, m, a, mp.nstr(v, 8), round(time.time()-t, 3))
