import sys, time
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts')
import mpmath as mp
import numpy as np
import norminvgauss_ref as R
from scipy import stats
mp.mp.dps = 100
rows = [  # method, arg, a, b, reported reference
 ('sf', -1e-10, 1e6, 5e5, '1.0'),
 ('sf', -707.10625086973585, 10, 9.99999, '1.0'),
 ('sf', 179088.24039927218, 10, 9.99999, '0.00017082512756882697'),
 ('cdf', 0.54012607482554176, 1e6, 5e5, '3.380440201163224e-201'),
 ('cdf', 0.57114623679561172, 1e6, 5e5, '2.77216473079964e-07'),
 ('cdf', -0.3, 1e4, 0, '7.784620646337754e-194'),
 ('cdf', 707.10625086973585, 10, 9.99999, '0.9178707888557534'),
 ('cdf', 179088.24039927218, 10, 9.99999, '0.9998291748724312'),
 ('cdf', -0.0001, 1e8, 0, '0.15865525332653024'),
 ('cdf', -0.001, 1e7, 0, '0.0007827018727833291'),
 ('cdf', -600, 1, 0, '1.9503586841127083e-265'),
 ('cdf', -30, 1, 0, '5.860566997067416e-16'),
 ('sf', 10, 1, 0, '1.3409847489184868e-06'),
 ('sf', 800, 1.25, 0.5, '2.1840732813829774e-265'),
 ('cdf', -6, 1.25, 0.5, '1.2357882365041296e-06'),
 ('sf', 9, 1.25, 0.5, '6.4886227458238e-05'),
 ('cdf', -4, 1, 0, '0.0017556126534699257'),
 ('cdf', 10, 1, 0, '0.9999986590152511'),
 ('isf', 0.29, 10, 9.99999, '67.44589577432565'),
 ('isf', 0.99, 10, 9.99999, '1.360732639133925'),
 ('isf', 0.57, 1e6, 5e5, '0.5771311001290594'),
 ('isf', 1e-10, 10, 9.99999, '1351130.1118913342'),
 ('isf', 0.15, 10, 9.99999, '240.42557163888097'),
 ('isf', 1e-300, 10, 9.99999, '67549018.8430328'),
 ('isf', 0.3, 1e8, 0, '5.244005109217989e-05'),
 ('isf', 0.49999999999999994, 1, 0, '1.0658744816094324e-16'),
 ('isf', 0.49999999949999741, 1, 0, '9.600593660978633e-10'),
 ('isf', 1e-10, 1.25, 0.5, '24.982927090751097'),
 ('isf', 1e-300, 1.25, 0.5, '908.2422209188078'),
 ('isf', 1e-300, 1e6, 5e5, '0.6237847906292711'),
]
only = sys.argv[1:]  # optional filter on b
for m, x, a, b, old in rows:
    if only and str(b) not in only: continue
    t = time.time()
    v = getattr(R, m)(mp.mpf(x), mp.mpf(a), mp.mpf(b))
    sc = float(getattr(stats.norminvgauss, m)(x, a, b))
    new = float(v)
    rel = abs(new - float(old)) / abs(float(old))
    ulp = abs(sc - v) / (abs(v) * 2.220446049250313e-16) if v != 0 else float('nan')
    print(f"{m}({x!r}, {a}, {b}) new={new!r} old={old} rel_change={rel:.2e} scipy={sc!r} ulp={float(ulp):.2g} [{time.time()-t:.0f}s]", flush=True)
