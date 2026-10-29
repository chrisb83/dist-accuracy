import warnings, numpy as np, mpmath as mp
from scipy import stats
warnings.simplefilter("ignore")
D = stats.genpareto

def logpdf_ref(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    t = c*x
    if abs(t) > 1:  # route: log(cx) + log1p(1/(cx))
        L = mp.log(c) + mp.log(x) + mp.log1p(1/t)
    else:
        L = mp.log1p(t)
    return -(1 + 1/c)*L

def logsf_ref(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    t = c*x
    if abs(t) > 1:
        L = mp.log(c) + mp.log(x) + mp.log1p(1/t)
    else:
        L = mp.log1p(t)
    return -L/c

def cdf_ref(x, c):
    # numerical integration of the docstring pdf
    x, c = mp.mpf(x), mp.mpf(c)
    f = lambda u: mp.exp(logpdf_ref(u, c)) if False else mp.exp(-(1+1/c)*mp.log1p(c*u))
    top = min(x, mp.mpf(800))
    pts = [mp.mpf(0)] + [p for p in (mp.mpf('1e-12'), mp.mpf('1e-6'), mp.mpf(1), mp.mpf(10), mp.mpf(100)) if p < top] + [top]
    return mp.quad(f, pts)

def logpdf_power(x, c):
    # independent route for F4: log of (1+c*x)**(-1-1/c)
    x, c = mp.mpf(x), mp.mpf(c)
    return mp.log(mp.power(1 + c*x, -1 - 1/c))

cases = [
 ("F1","cdf",cdf_ref,2.0,1e-300),("F1","cdf",cdf_ref,5e145,1e-300),("F1","cdf",cdf_ref,1.0,1e-160),
 ("F1","cdf",cdf_ref,0.001,1e-152),("F1","cdf",cdf_ref,1e-10,1e-300),
 ("F2","logpdf",logpdf_ref,1e-30,1e-300),("F2","logpdf",logpdf_ref,2.4703282292062327e-164,-1e-160),
 ("F2","logsf",logsf_ref,1e-20,1e-300),("F2","logpdf",logpdf_ref,4.89124989382834e-24,1e-300),
 ("F3","logsf",logsf_ref,1e300,1e10),("F3","logpdf",logpdf_ref,1e300,1e10),("F3","logpdf",logpdf_ref,1e308,1000.),
 ("F4","logpdf",logpdf_ref,1e160,-1e-160),("F4","logpdf",logpdf_power,1e160,-1e-160),("F4","logsf",logsf_ref,1e160,-1e-160),
]
for fid, meth, ref, x, c in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(ref(x, c))
    mp.mp.dps = 200
    agree = mp.almosteq(vals[0], vals[1], rel_eps=mp.mpf(10)**-30) if vals[1] != 0 else vals[0]==0
    r = vals[1]
    s = getattr(D, meth)(x, c)
    ulp = float(abs(mp.mpf(s) - r)/abs(r)/mp.mpf(2)**-52) if np.isfinite(s) else float('inf')
    print(f"{fid} {meth}({x!r}, {c!r}) scipy={s!r} ref={mp.nstr(r,17)} agree30={agree} ulp={ulp:.3g}  [{ref.__name__}]")

# logcdf note and inv_boxcox1p shortcut
from scipy import special as sc
mp.mp.dps=100
print("logcdf(0.5,1e-200)", D.logcdf(0.5,1e-200), mp.nstr(mp.log(cdf_ref(0.5,1e-200)),17))
print("inv_boxcox1p(-2,-1e-300)", sc.inv_boxcox1p(-2.,-1e-300), "inv_boxcox1p(-2,-1e-150)", sc.inv_boxcox1p(-2.,-1e-150))
print("c*x products:", 1e-300*1e-30, -1e-160*2.4703282292062327e-164, 1e10*1e300, -1e-160*1e160)
