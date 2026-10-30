"""Check that the reference pdf integrates to 1 (30 digits) for every PARAMS set."""
import sys
import mpmath as mp
sys.path.insert(0, r"C:\Users\chbau\Documents\research\dist-accuracy\refs")
sys.path.insert(0, r"C:\Users\chbau\Documents\research\dist-accuracy\.claude\skills\dist-accuracy-review\scripts")
import gausshyper_ref as R

mp.mp.dps = 40
for s in R.PARAMS:
    P = [mp.mpf(v) for v in s]
    a, b = float(s[0]), float(s[1])
    m = a / (a + b)
    pts = sorted(set([0, max(m - 0.2, 1e-9), m, min(m + 0.2, 1 - 1e-9), 1]))
    if s[3] < -0.99:
        pts = sorted(set(pts + [0.99, 0.9999]))
    I = mp.quad(lambda t: R.pdf(t, *P), pts, maxdegree=10)
    print(s, mp.nstr(I - 1, 5))
