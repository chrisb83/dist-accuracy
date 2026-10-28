import sys, time, mpmath as mp
sys.path.insert(0,'refs')
import kappa4_ref as r
P=[mp.mpf(0.1),mp.mpf(0.0)]
X=[mp.mpf(-0.7211113395855918),mp.mpf(0.4009700988683587),mp.mpf(2.255630727750341)]
with mp.workdps(30):
    for x1,x2 in [(X[0],X[1]),(X[1],X[2])]:
        t0=time.time()
        I=mp.quad(lambda t: r.pdf(t,*P), [x1,x2])
        print(mp.nstr(I,10), '%.2f s'%(time.time()-t0), flush=True)
