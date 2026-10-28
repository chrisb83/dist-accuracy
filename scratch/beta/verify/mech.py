import mpmath as mp, numpy as np, scipy.special as sc, scipy.special._ufuncs as scu
mp.mp.dps=60
for a,b in ((1e6,2),(1e5,3e5)):
    t=mp.loggamma(a)+mp.loggamma(b)-mp.loggamma(a+b); v=sc.betaln(a,b)
    print('betaln',a,b,v,'abs err',mp.nstr(abs(mp.mpf(v)-t),3))
print('betainccinv', sc.betainccinv(1e6,2,1e-200), sc.betainccinv(20,50,1e-300), '_beta_ppf mirror', 1-scu._beta_ppf(1e-300,50,20), scu._beta_ppf(1e-300,50,20))
print('betaincinv',sc.betaincinv(0.5,2,1e-9))
for x in (1e-310,):
    try: print(scu._beta_pdf(x,0.5,5))
    except Exception as e: print('raw _beta_pdf', type(e).__name__, e)
try: print(scu._beta_pdf(1e-310,2.31,0.627))
except Exception as e: print('b<1', type(e).__name__)
