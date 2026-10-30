import sys, time, warnings
sys.path.insert(0, 'refs'); sys.path.insert(0, '.claude/skills/dist-accuracy-review/scripts')
import mpmath as mp
mp.mp.dps = 100
import norminvgauss_ref as r
from scipy import stats
warnings.simplefilter('ignore')
M = mp.mpf
for (q, a, b) in [(0.3, 1e-170, 0.0), (0.3, 1e-170, 5e-171)]:
    t = time.time()
    ref = r.isf(M(q), M(a), M(b))
    try:
        s = stats.norminvgauss.isf(q, a, b)
    except Exception as e:
        s = repr(e)
    print(f"isf({q}, {a}, {b}): scipy {s}  reference {mp.nstr(ref, 17)}  ({time.time()-t:.1f} s)")
    print("  mean", stats.norminvgauss.mean(a, b), " cdf(1)", stats.norminvgauss.cdf(1.0, a, b),
          "ref", mp.nstr(r.cdf(M(1), M(a), M(b)), 17))
