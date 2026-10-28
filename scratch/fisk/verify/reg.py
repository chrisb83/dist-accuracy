import numpy as np, warnings
from scipy.stats import fisk
warnings.simplefilter('ignore')
for c in [2.0, 3.085754862225318, 0.5, 1e-3]:
    bad=[]
    for e in range(-308, 0):
        v = fisk.pdf(10.0**e, c)
        if not np.isfinite(v) : bad.append((e, v))
    if bad:
        print(f"c={c}: non-finite pdf for exponents {bad[0][0]} .. {bad[-1][0]}, kinds={set(str(b[1]) for b in bad)}")
    else:
        print(f"c={c}: all finite")
