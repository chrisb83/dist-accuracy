import sys, time, warnings
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/rdist/verify')
import mpmath as mp, numpy as np, scipy.stats as st
import vref
warnings.simplefilter('ignore')
calls = [
 ('F1','ppf','0.4999999999999999','3'),('F1','ppf','0.5000000000000001','1.6'),
 ('F1','ppf','0.4999999999999999','0.001'),('F1','ppf','0.49999999999999','1e6'),
 ('F1','ppf','0.500000000000001','1000'),('F1','ppf','0.499999999999999','1000'),
 ('F1','ppf','0.49999999999999','3'),('F1','ppf','0.500001','1e6'),
 ('F1','ppf','0.4999','1e6'),('F1','ppf','0.4999','10'),('F1','ppf','0.6','1e6'),
 ('F1','ppf','0.4','1e6'),('F1','ppf','0.6','1000'),
 ('F2','sf','0.9999999999999999','0.001'),('F2','sf','0.9999999999999999','0.5'),
 ('F2','sf','0.9999999999999999','3'),('F2','pdf','0.9999999999999999','1.6'),
 ('F2','pdf','0.9999999999999999','3'),('F2','logpdf','0.9999999999999999','1000'),
 ('F2','logpdf','0.9999999999999999','0.001'),('F2','cdf','0.9999999999999999','0.001'),
 ('F2','cdf','0.9999999999999999','0.5'),
 ('F3','pdf','0.004999997500001875','1e6'),('F3','pdf','0.0','1e6'),
 ('F3','pdf','0.029999985000011248','1e6'),('F3','pdf','0.0','1000'),
 ('F3','pdf','-0.8','1000'),('F3','logpdf','0.00199999900000075','1e6'),
 ('F3','logpdf','0.0','1e6'),('F3','logpdf','0.0632139541241014','1000'),
 ('F4','cdf','-0.029999985000011248','1e6'),('F4','sf','0.029999985000011248','1e6'),
 ('F4','cdf','-0.0199999900000075','1e6'),('F4','cdf','-0.00199999900000075','1e6'),
 ('F4','cdf','3e-09','1e8'),('F4','sf','3e-09','1e8'),
]
only = sys.argv[1:] 
for fid, m, a, c in calls:
    if only and fid not in only: continue
    t0=time.time()
    vals=[]
    for d in (100,200):
        mp.mp.dps=d
        vals.append(getattr(vref,m)(a,c))
    mp.mp.dps=200
    agree = (vals[0]==vals[1]==0) or abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref=vals[1]
    s = getattr(st.rdist, m)(float(a), float(c))
    extra=''
    if m=='ppf':
        # check cdf(x_ref) == q
        r = vref.cdf.__wrapped__ if hasattr(vref.cdf,'__wrapped__') else None
        x=ref; cc=mp.mpf(float(c))
        F = (1 - mp.betainc(cc/2, mp.mpf(1)/2, 0, 1-x*x, regularized=True)/2) if x>0 else mp.betainc(cc/2, mp.mpf(1)/2, 0, 1-x*x, regularized=True)/2
        extra=' cdfchk=%s' % mp.nstr(abs(F-mp.mpf(float(a))),3)
    if ref==0: ulp=float('nan')
    else:
        ulp = float(abs(mp.mpf(s)-ref)/abs(ref)/mp.mpf(2)**-52) if np.isfinite(s) else float('inf')
    print(f"{fid} {m}({a}, {c}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree30={agree}{extra} t={time.time()-t0:.1f}s", flush=True)
