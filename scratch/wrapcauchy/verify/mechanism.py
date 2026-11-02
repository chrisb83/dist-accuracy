# Isolate numerator vs denominator contributions
import numpy as np, mpmath as mp
mp.mp.dps = 100
def r(x, c):
    x = mp.mpf(x); c = mp.mpf(c); return (1-c*c)/(2*mp.pi*(1+c*c-2*c*mp.cos(x)))
for x, c in [(3.141592653589793, 0.9999999870854271), (2.0, 0.9999999870854271), (0.0, 0.999999), (1e-6, 0.9999999999)]:
    ref = r(x, c)
    num_old = 1.0 - c*c; num_new = (1-c)*(1+c)
    den_old = 1 + c*c - 2*c*np.cos(x); den_new = (1-c)**2 + 4*c*np.sin(x/2)**2
    for name, n, d in [("old/old", num_old, den_old), ("newnum/olddEN", num_new, den_old), ("oldnum/newden", num_old, den_new), ("new/new", num_new, den_new)]:
        v = n/(2*np.pi*d)
        print(x, c, name, f"{float(abs(v-ref)/ref)/2.22e-16:.3g} ulp")
