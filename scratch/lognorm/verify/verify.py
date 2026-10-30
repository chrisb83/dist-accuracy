import warnings; warnings.filterwarnings("ignore")
import mpmath as mp, numpy as np
from scipy import stats
# docstring: f(x,s) = exp(-log(x)^2/(2 s^2)) / (s x sqrt(2 pi))
# Route A (logpdf): log of normal density npdf(z) via mp.npdf, z=L/s, minus L and log s.
# Route B (pdf): derivative of ncdf(L/s) w.r.t. L (mp.diff), divided by x.
def logpdf_ref(x, s):
    x, s = mp.mpf(x), mp.mpf(s); L = mp.log(x); z = L/s
    return mp.log(mp.npdf(z)) - L - mp.log(s)
def pdf_ref(x, s):
    x, s = mp.mpf(x), mp.mpf(s); L0 = mp.log(x)
    d = mp.diff(lambda L: mp.ncdf(L/s), L0)
    return d / x
cases = [("logpdf",1.7e308,1),("logpdf",1.7e308,0.9536822696057533),("logpdf",1e307,10),
 ("logpdf",1e305,1000),("logpdf",1e300,1e10),("logpdf",8e307,1),("logpdf",7e307,1),
 ("logpdf",1e-308,1e-17),("logpdf",1e-300,1e-150),("logpdf",1e-180,1e-150),
 ("pdf",1.0,1e-200),("logpdf",1.0,1e-200),("logpdf",1.000001,1e-160),("logpdf",1.0000001,1e-158),
 ("logpdf",3.660042966917285e-86,10),("logpdf",3.6600466269602515e-86,10),
 ("pdf",1e-210,10),("pdf",1e-170,10)]
for m,x,s in cases:
    f = logpdf_ref if m=="logpdf" else pdf_ref
    vals=[]
    for d in (100,200):
        mp.mp.dps=d; vals.append(f(x,s))
    mp.mp.dps=200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30 or vals[1]==0
    ref = vals[1]
    sp = float(getattr(stats.lognorm, m)(x, s))
    if np.isfinite(sp):
        ulp = float(abs(mp.mpf(sp)-ref)/(abs(ref)*mp.mpf(2)**-52))
    else: ulp = float('inf')
    print(f"{m}({x!r}, {s!r}) scipy={sp!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree30={agree}")
print("--- pdf(1,1e-200) via exp(route A) and 1/(s sqrt(2pi))")
for d in (100,200):
    mp.mp.dps=d; print(mp.nstr(mp.exp(logpdf_ref(1.0,1e-200)),20), mp.nstr(1/(mp.mpf(1e-200)*mp.sqrt(2*mp.pi)),20))
print(np.float64(1.57e-162)**2, np.float64(1.58e-162)**2)
