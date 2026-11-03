import sys, time, warnings
sys.path.insert(0, ".claude/skills/dist-accuracy-review/scripts"); sys.path.insert(0, "refs")
import mpmath as mp, numpy as np
import genhyperbolic_ref as R
from scipy.stats import genhyperbolic as G
warnings.simplefilter("ignore")
mp.mp.dps = 100
for sh in R.PARAMS:
    P = [mp.mpf(s) for s in sh]
    for x in [-2.0, 0.0, 1.5]:
        t=time.time(); c = R.cdf(mp.mpf(x), *P); tc=time.time()-t
        t=time.time(); s = R.sf(mp.mpf(x), *P); ts=time.time()-t
        t=time.time(); f = R.pdf(mp.mpf(x), *P); tp=time.time()-t
        print(sh, x, "pdf", mp.nstr(f,12), G.pdf(x,*sh), f"{tp:.2f}s", "cdf", mp.nstr(c,12), G.cdf(x,*sh), f"{tc:.2f}s",
              "c+s-1", mp.nstr(c+s-1,3), f"{ts:.2f}s")
# cdf vs quad of pdf
mp.mp.dps = 40
P = [mp.mpf(s) for s in (0.5,1.5,-0.5)]
for x in [-20, -3, 0.5]:
    q = mp.quad(lambda t: R.pdf(t,*P), [-mp.inf, x])
    print("quad", x, mp.nstr(q, 25), mp.nstr(R.cdf(mp.mpf(x),*P), 25))
