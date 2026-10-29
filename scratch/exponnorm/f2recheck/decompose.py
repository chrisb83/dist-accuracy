# Split the F2 cdf error into the part from ndtr(x) and the part from the subtracted term
import numpy as np, scipy.special as sc, mpmath as mp, scipy.stats as st
mp.mp.dps = 100
eps = 2.220446049250313e-16
pts = [(-37.0,1e4),(-20.0,1e8),(-30.0,10.0),(-37.0,1.0),(-37.0,1.5),(-37.0,0.999),(-37.0,0.1),(-10.0,1.5),
       (-30.3,1.5),(-20.7,1.5),(-30.3,0.1),(-20.7,0.1),(-36.0,1.5),(-33.0,1.5),(-28.0,1.5)]
print("x K | F_ulp | ndtr_ulp(own) | ndtr err in F-ulp | term2_ulp(own) | term2 err in F-ulp | F_ulp with exact ndtr | Phi/F")
for x,K in pts:
    X=mp.mpf(x); KK=mp.mpf(K)
    r1 = mp.erfc(-X/mp.sqrt(2))/2
    r2 = mp.exp(1/(2*KK**2)-X/KK)*mp.erfc(-(X-1/KK)/mp.sqrt(2))/2
    F = r1-r2
    t1 = sc.ndtr(x)
    u = (-x+1.0/K)/np.sqrt(2)
    if u>=0 and K<1:
        t2 = 0.5*np.exp(-0.5*x**2)*sc.erfcx(u)
    else:
        invK=1.0/K; t2=np.exp(invK*(0.5*invK-x)+sc.log_ndtr(x-invK))
    Fs = float(st.exponnorm.cdf(x,K))
    e1 = mp.mpf(t1)-r1; e2 = mp.mpf(t2)-r2
    Fexact1 = float(r1) - t2   # correctly rounded ndtr
    f = lambda v: float(abs(v)/F/eps)
    print(f"{x} {K:g} | {f(mp.mpf(Fs)-F):.3g} | {float(abs(e1)/r1/eps):.3g} | {f(e1):.3g} | {float(abs(e2)/r2/eps):.3g} | {f(e2):.3g} | {f(mp.mpf(Fexact1)-F):.3g} | {float(r1/F):.3g}")
