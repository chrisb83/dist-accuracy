# ndtr / erfc / log_ndtr alone vs mpmath in the far left tail
import numpy as np, scipy.special as sc, mpmath as mp
mp.mp.dps = 60
eps = 2.220446049250313e-16
def ulp(a, r): return float(abs(mp.mpf(a) - r) / abs(r) / eps)
xs = list(np.arange(-38.0, -4.0, 1.0)) + [-37.5, -30.3, -20.7, -10.1, -5.5]
print("x      ndtr_ulp  erfc(-x/s2)/2_ulp  erfcx(-x/s2)_ulp  log_ndtr_ulp  cond_ndtr")
for x in sorted(xs):
    X = mp.mpf(float(x))
    r = mp.erfc(-X / mp.sqrt(2)) / 2
    rx = mp.erfcx(-X / mp.sqrt(2)) if hasattr(mp,'erfcx') else mp.exp(X**2/2)*mp.erfc(-X/mp.sqrt(2))
    a = sc.ndtr(x); b = 0.5*sc.erfc(-x/np.sqrt(2)); c = sc.erfcx(-x/np.sqrt(2))
    d = sc.log_ndtr(x)
    pdf = mp.npdf(X)
    print(f"{x:6.1f} {ulp(a,r):9.3g} {ulp(b,r):12.3g} {ulp(c,rx):14.3g} {ulp(d,mp.log(r)):12.3g} {float(abs(X)*pdf/r):10.3g}")
