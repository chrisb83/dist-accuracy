import sys, numpy as np, warnings, mpmath as mp
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts')
import rel_breitwigner_ref as R
from scipy import stats
mp.mp.dps = 100
M = np.finfo(float).max; tiny = np.finfo(float).tiny
print("analytic bound DBL_MAX*sqrt(tiny*pi/2) =", M*np.sqrt(tiny*np.pi/2))
ov = lambda x, rho: (x-rho)*(x+rho) == np.inf
for rho in [1.6e154, 2e154, 3e154, 3.3e154, 3.35e154, 3.37e154, 3.4e154, 4e154, 1e155, 1e160]:
    lo, hi = rho, 2*rho   # ov(lo) False, ov(hi) True
    assert not ov(lo, rho) and ov(hi, rho)
    for _ in range(200):
        mid = lo + (hi-lo)/2
        if mid in (lo, hi): break
        lo, hi = (lo, mid) if ov(mid, rho) else (mid, hi)
    x = hi  # first overflowing x above rho
    s = stats.rel_breitwigner.pdf(x, rho); sl = stats.rel_breitwigner.pdf(lo, rho)
    t = R.pdf(x, rho)
    print(f"rho={rho:.4g} x_ov={x!r} scipy={s} (at prev double {sl:.3g}) true={mp.nstr(t,6)} normal={t>tiny}")
