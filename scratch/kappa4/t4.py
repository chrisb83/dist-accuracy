import sys, time, mpmath as mp
sys.path.insert(0,'refs')
import kappa4_ref as r
mp.mp.dps=400
tests = [('cdf',-1e300,0,0),('pdf',-1e300,0,0),('logcdf',-1e300,0,0),('logpdf',-1e300,0,0),
         ('sf',-1e300,0,0),('logsf',-1e300,0,0),('cdf',1e300,0,0),('sf',1e300,0,0),
         ('cdf',-1e300,-1e6,0),('logcdf',-1e300,-1e6,0),('cdf',-1e8,-1e6,0),('logcdf',-1e8,-1e6,0),
         ('cdf',-1e35,-1e3,0.1),('pdf',-1e35,-1e3,0.1),('cdf',-1e300,-1e3,0.1),
         ('ppf',1e-300,-2,-0.5),('ppf',1e-300,-1e6,0.0),('isf',1e-300,-1e6,0.0),
         ('cdf',1.0,0.5,1e-8),('ppf',0.3,1e-8,0.5),('cdf',0.0,-1,0),('ppf',0.5,-1,0)]
for m,x,h,k in tests:
    t0=time.time()
    try:
        v=r.REFS[m](mp.mpf(x),mp.mpf(h),mp.mpf(k)); s=mp.nstr(v,12)
    except Exception as e: s='ERR '+repr(e)
    print('%-7s x=%-9g h=%-8g k=%-6g %-28s %.2fs'%(m,x,h,k,s,time.time()-t0), flush=True)
