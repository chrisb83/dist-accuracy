import numpy as np, warnings
from scipy import special as sc, integrate
from scipy.stats import genhyperbolic as g
print("min(1.0, nan) ->", min(1.0, float('nan')), " max(0.0, min(1.0, nan)) ->", max(0.0, min(1.0, float('nan'))))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("cdf(0,-1,1,1)", g.cdf(0.0, -1, 1, 1), [str(x.message) for x in w])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("cdf(0,100,.01,0)", g.cdf(0.0, 100, .01, 0), "sf", g.sf(0.0, 100, .01, 0), [str(x.message)[:60] for x in w])
p,a,b = -1.,1.,1.
d = np.sqrt((a+b)*(a-b))
with np.errstate(all='ignore'):
    print("d", d, "kv(p+1,d)", sc.kv(p+1,d), "kv(p,d)", sc.kv(p,d), "mean", b/d*sc.kv(p+1,d)/sc.kv(p,d))
print("kve(100,0.01)", sc.kve(100, 0.01), "kve(99.5, .01)", sc.kve(99.5, .01))
print("kve(0.5, 2**30-0.5)", sc.kve(0.5, 2.0**30-0.5), sc.kve(0.5, 2.0**30-1))
# overflow threshold in p for g = 0.01, 1, 10
for gg in [0.01, 1.0, 10.0, 100.]:
    ps = np.arange(1, 400)
    with np.errstate(all='ignore'):
        bad = ps[~np.isfinite(sc.kve(ps, gg))]
    print("g", gg, "first p with kve(p,g) not finite:", bad[:1])
