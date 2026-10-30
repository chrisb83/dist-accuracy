# Independent reference: normalization by mp.quad of the unnormalized
# docstring density (no hyp2f1, no beta), pdf in log space.
import sys, time
import mpmath as mp
import numpy as np
from scipy import stats

CASES = [
 ("F1", 0.05, 300, 300, 5, 2),
 ("F1", 0.9500000000000001, 300, 300, 5, 2),
 ("F2", 0.5, 600, 600, 1, 1),
 ("F2", 0.5, 1000, 1000, 0, 0.5),
 ("F3", 0.9, 2, 3, -120, 1000),
 ("F3", 1e-100, 3, 2, 1e4, 1),
 ("F3", 0.0001, 2, 3, 1000, 10),
 ("F3", 0.5, 3, 2, 1e5, 1e-5),
 ("F4", 0.5, 2.5, 1.5, 2.500000000001, 1e6),
 ("F4", 0.5, 2.5, 1.5, 2.5, 1e10),
 ("F5", 0.5, 3, 2, 1e4, 1e-4),
 ("F6", 1e-210, 2.5, 1.5, 2.5, 1e6),
 ("F7", 1e-06, 2, 1e6, 3, 0.5),
 ("F7", 1e-300, 2, 1e4, 3, 0.5),
]

def logf(t, a, b, c, z):
    return (a-1)*mp.log(t) + (b-1)*mp.log1p(-t) - c*mp.log1p(z*t)

def lognorm(a, b, c, z):
    # find the mode of the integrand on a log grid to scale it
    pts = [mp.mpf(10)**(-k) for k in range(30, 0, -1)]
    pts += [mp.mpf(i)/40 for i in range(4, 40)]
    pts += [1 - mp.mpf(10)**(-k) for k in range(2, 30)]
    pts = sorted(set(pts))
    m = max(logf(p, a, b, c, z) for p in pts)
    g = lambda t: mp.exp(logf(t, a, b, c, z) - m) if 0 < t < 1 else mp.mpf(0)
    I = mp.quad(g, [0] + pts + [1])
    return mp.log(I) + m

def ref(x, a, b, c, z, dps):
    mp.mp.dps = dps
    x, a, b, c, z = map(mp.mpf, (x, a, b, c, z))
    return mp.exp(logf(x, a, b, c, z) - lognorm(a, b, c, z))

for fid, x, a, b, c, z in CASES:
    t = time.time()
    r1 = ref(x, a, b, c, z, 100)
    r2 = ref(x, a, b, c, z, 200)
    mp.mp.dps = 200
    agree = -mp.log10(abs(r1/r2 - 1)) if r1 != r2 else 200
    with np.errstate(all='ignore'):
        s = stats.gausshyper.pdf(x, a, b, c, z)
    rf = float(r2)
    if np.isfinite(s) and rf != 0:
        ulp = abs(mp.mpf(s) - r2)/(abs(r2)*mp.mpf(2)**-52)
        ulp = float(ulp)
    else:
        ulp = float('inf')
    print(f"{fid} pdf({x!r}, {a}, {b}, {c}, {z}) scipy={s!r} ref={mp.nstr(r2, 17)} "
          f"ulp={ulp:.3g} agree100/200={float(agree):.1f}d t={time.time()-t:.1f}s", flush=True)
