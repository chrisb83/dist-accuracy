import sys, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts')
import mpmath as mp
import norminvgauss_ref as R
from scipy import stats
mp.mp.dps = 100
for q, a, b in [(0.3, 1e6, 5e5), (0.5, 1e6, 5e5), (0.3, 1e8, 0.0), (0.71, 10, 9.99999), (1e-10, 1, 0)]:
    try:
        sc = stats.norminvgauss.ppf(q, a, b)
    except Exception as e:
        sc = repr(e)
    ref = R.isf(1 - mp.mpf(q), mp.mpf(a), mp.mpf(b))
    print(q, a, b, 'scipy', sc, 'ref', mp.nstr(ref, 17), flush=True)
