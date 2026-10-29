# per-term errors of the erfcx-branch cdf at x=-37, K=0.1 and default branch at K=1.5
import numpy as np, scipy.special as sc, mpmath as mp
mp.mp.dps = 100
x, K = -37.0, 0.1
u = (-x + 1.0/K)/np.sqrt(2)
t1 = sc.ndtr(x); t2 = 0.5*np.exp(-0.5*x**2)*sc.erfcx(u)
X = mp.mpf(x); KK = mp.mpf(K)
r1 = mp.erfc(-X/mp.sqrt(2))/2
r2 = mp.exp(1/(2*KK**2) - X/KK)*mp.erfc(-(X - 1/KK)/mp.sqrt(2))/2
eps = 2.220446049250313e-16
print("K=0.1 ndtr ulp", mp.nstr(abs(t1-r1)/r1/eps,3), "term2 ulp", mp.nstr(abs(t2-r2)/r2/eps,3), "ratio Phi/F", mp.nstr(r1/(r1-r2),4))
K = 1.5; KK = mp.mpf(K); invK = 1/K
lp = invK*(0.5*invK - x) + sc.log_ndtr(x - invK); t2 = np.exp(lp)
r2 = mp.exp(1/(2*KK**2) - X/KK)*mp.erfc(-(X - 1/KK)/mp.sqrt(2))/2
print("K=1.5 ndtr ulp", mp.nstr(abs(sc.ndtr(x)-r1)/r1/eps,3), "exp(logprod) ulp", mp.nstr(abs(t2-r2)/r2/eps,3), "ratio Phi/F", mp.nstr(r1/(r1-r2),4))
