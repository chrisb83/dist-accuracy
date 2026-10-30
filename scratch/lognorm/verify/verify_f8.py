import mpmath as mp, numpy as np, warnings
from scipy import stats
import scipy
warnings.simplefilter("ignore")
pts = [("logpdf",5e-324,30),("logpdf",5e-324,1),("logpdf",1e-320,10),("logpdf",1e-320,1000),
       ("logpdf",1e-315,30),("pdf",5e-324,30),("pdf",1e-322,100),("pdf",1e-315,100)]
def ref(m, x, s, dps):
    with mp.workdps(dps):
        x = mp.mpf(x); s = mp.mpf(s); L = mp.log(x)
        # route: derivative of cdf ncdf(L/s) w.r.t. L, divided by x
        d = mp.diff(lambda t: mp.ncdf(t/s), L)
        p = d / x
        # cross-check: closed form via npdf
        p2 = mp.npdf(L/s)/(s*x)
        assert abs(p/p2-1) < mp.mpf(10)**(-dps//2), (p, p2)
        return mp.log(p) if m == "logpdf" else p
print(scipy.__version__)
for m,x,s in pts:
    r1 = ref(m,x,s,100); r2 = ref(m,x,s,200)
    agree = abs(r1/r2-1) < mp.mpf(10)**-30
    sv = getattr(stats.lognorm, m)(x, s)
    ulp = abs((mp.mpf(sv)-r2)/r2)/mp.mpf(2.220446049250313e-16)
    print(m, x, s, repr(sv), mp.nstr(r2, 17), mp.nstr(ulp, 3), agree)
