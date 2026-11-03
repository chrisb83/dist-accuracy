"""Candidate logpdf: xlog1py(c/2-1, -x*x) - (0.5*log(pi) - log(poch(c/2, 0.5)))."""
import mpmath as mp, numpy as np, scipy.special as sc
from scipy import stats
mp.mp.dps = 60
def ref(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return (c/2-1)*mp.log(1-x*x) - (mp.loggamma(0.5)+mp.loggamma(c/2)-mp.loggamma(c/2+0.5))
def ulp(v, r): return float(abs((mp.mpf(v)-r)/r))/2.22e-16
for c, x in [(1e3, 0.0), (1e4, 0.0), (1e6, 0.0), (1e6, 0.001), (1e6, 0.01), (1e8, 0.0), (1e3, 0.3), (10, 0.0)]:
    r = ref(x, c)
    cand = sc.xlog1py(c/2-1, -x*x) - (0.5*np.log(np.pi) - np.log(sc.poch(c/2, 0.5)))
    print(f"c={c:g} x={x:g}: scipy {ulp(stats.rdist.logpdf(x, c), r):.3g} ulp, candidate {ulp(cand, r):.3g} ulp;"
          f" pdf scipy {ulp(stats.rdist.pdf(x, c), mp.exp(r)):.3g} ulp")
