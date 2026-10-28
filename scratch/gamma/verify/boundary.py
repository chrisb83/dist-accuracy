import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
import scipy.special as sc
from quadref import P_quad
EPS = 2.220446049250313e-16
for a in (1e6,):
    for k in (3.0, 4.0, 4.4, 4.49, 4.51, 4.6, 5.0, 8.0, 12.0, 15.0, 20.0, 30.0):
        mp.dps = 60
        x = float(mp.mpf(a) - k*mp.sqrt(mp.mpf(a)))
        r = P_quad(a, x, dps=100); mp.dps = 100
        s = sc.gammainc(a, x)
        print("a=%g  k=%5.2f x=%.17g  scipy=%.17g ref=%s relerr=%.3g ulp=%.3g"
              % (a, k, x, s, mp.nstr(r, 18), float(abs(mp.mpf(s)-r)/r), float(abs(mp.mpf(s)-r)/(r*EPS))))
