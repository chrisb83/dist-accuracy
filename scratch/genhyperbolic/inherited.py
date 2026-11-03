import sys, os, warnings, time
os.environ["GH_INHERITED"] = "1"
sys.path[:0] = ['.claude/skills/dist-accuracy-review/scripts', 'refs']
import mpmath as mp, genhyperbolic_ref as R
from scipy.stats import genhyperbolic as G
warnings.simplefilter("ignore")
mp.mp.dps = 100
sh = (0.5, 1.5, -0.5); P = [mp.mpf(v) for v in sh]
def row(m, x):
    t = time.time()
    r = float(R.REFS[m](mp.mpf(x), *P)); s = float(getattr(G, m)(x, *sh))
    print(f"| `{m}({x:g}, 0.5, 1.5, -0.5)` | {s!r} | {r!r} | {abs(s-r)/abs(r)/2.22e-16:.2g} | {time.time()-t:.0f}s", flush=True)
row("logsf", 600.0)
row("logcdf", -800.0)
row("ppf", 1e-30)
row("isf", 1e-30)
row("ppf", 0.3)
