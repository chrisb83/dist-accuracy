import sys, warnings
sys.path[:0] = ['.claude/skills/dist-accuracy-review/scripts', 'refs']
import mpmath as mp, numpy as np, genhyperbolic_ref as R
from scipy.stats import genhyperbolic as G
warnings.simplefilter("ignore")
mp.mp.dps = 100
def row(m, x, *sh):
    P = [mp.mpf(v) for v in sh]
    try: r = R.REFS[m](mp.mpf(x), *P); rf = float(r)
    except Exception as e: rf = float('nan')
    s = float(getattr(G, m)(x, *sh))
    ulp = abs(s - rf) / abs(rf) / 2.22e-16 if rf else float('nan')
    print(f"| `{m}({x:g}, {', '.join(f'{v:g}' for v in sh)})` | {s!r} | {rf!r} | {ulp:.2g} |", flush=True)
for m in ["pdf", "logpdf"]:
    row(m, 0.0, -1.0, 1.0, 1.0); row(m, 1.5, -1.0, 1.0, 1.0)
for m in ["cdf", "sf"]:
    row(m, 0.0, -1.0, 1.0, 1.0); row(m, -2.0, -1.0, 1.0, 1.0); row(m, 10.0, -1.0, 1.0, 1.0)
for m in ["pdf", "logpdf", "cdf", "sf"]:
    row(m, 0.0, 100.0, 0.01, 0.0); row(m, 3.0, 100.0, 0.01, 0.0)
row("pdf", 0.0, 80.0, 0.01, 0.0); row("pdf", 0.0, 81.0, 0.01, 0.0); row("pdf", 0.0, -100.0, 0.01, 0.0)
row("pdf", 0.0, 1.0, 1.1e9, 0.0); row("logpdf", 0.0, 1.0, 1.1e9, 0.0); row("pdf", 0.0, 1.0, 1e9, 0.0)
row("cdf", 0.0, 1.0, 1.1e9, 0.0)
