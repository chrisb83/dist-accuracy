import numpy as np, warnings, time
from scipy.stats import genhyperbolic as g
from scipy.stats import _stats
warnings.simplefilter("ignore")
def show(name, f, *args):
    t=time.time()
    try: v=f(*args)
    except Exception as e: v=repr(e)
    print(f"{name}{args} = {v!r}  ({time.time()-t:.2f}s)")
for x in [0., 1., 1e8, 7e8, 7.2e8, 1e9, 1e20, 1e160]:
    show("logpdf", g._logpdf, x, 0.5, 1.5, -0.5)
for x in [1e3, 1e6, 1e9, 1e12]:
    show("pdf b=a p<0", g._pdf, x, -1.0, 1.0, 1.0)
    show("pdf b~a", g._pdf, x, -1.0, 1.0, 0.999999)
show("pdf p=100 a=.01", g._pdf, 0.0, 100., 0.01, 0.)
show("pdf p=-100 a=.01", g._pdf, 0.0, -100., 0.01, 0.)
show("pdf a=1e6", g._pdf, 0.0, 1., 1e6, 0.)
show("pdf a=2e9", g._pdf, 0.0, 1., 2e9, 0.)
show("pdf a=1e155", g._pdf, 0.0, 1., 1e155, 0.)
for x in [-1e3,-30,-5,0,5,30,1e3]:
    show("cdf", g._cdf, x, 0.5, 1.5, -0.5); show("sf", g._sf, x, 0.5, 1.5, -0.5)
show("cdf b=a", g._cdf, 0.0, -1.0, 1.0, 1.0)
show("cdf b=a", g._cdf, -1e3, -1.0, 1.0, 1.0)
show("sf b=a", g._sf, 1e3, -1.0, 1.0, 1.0)
