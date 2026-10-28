import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
import scipy.special as sc
from crosscheck import P_series, P_quad2
EPS = 2.220446049250313e-16
for a in (1e8,):
    for k in (12.0, 15.0, 20.0, 30.0, 50.0):
        mp.dps = 60
        x = float(mp.mpf(a) - k*mp.sqrt(mp.mpf(a)))
        ser, n = P_series(a, x, dps=25)
        q = P_quad2(a, x, dps=60)
        mp.dps = 25
        s = sc.gammainc(a, x)
        print("a=%g k=%4.1f x=%.17g nterm=%d scipy=%.17g ser=%s relerr=%.3g (quad agrees %.2g)"
              % (a, k, x, n, s, mp.nstr(ser,16), float(abs(mp.mpf(s)-ser)/ser), float(abs(q-ser)/ser)))
