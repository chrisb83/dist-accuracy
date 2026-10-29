# Independent verifier reference for genhalflogistic (standard form).
# pdf f(x,c) = 2 (1-cx)^(1/c-1) / (1+(1-cx)^(1/c))^2  (docstring writes 1/(c-1): typo;
# derivative of the cdf (1-t)/(1+t), t=(1-cx)^(1/c), gives 1/c-1, matching the code comment).
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
D = stats.genhalflogistic

def pdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    L = mp.log1p(-c*x)          # log(1-cx)
    return 2*mp.exp((1/c-1)*L) / (1+mp.exp(L/c))**2

def cdf_quad(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    # integrate pdf from 0 to x; split interval for accuracy
    return mp.quad(lambda t: pdf(t, c), mp.linspace(0, x, 5))

def ppf(q, c):
    # solve (1-t)/(1+t)=q -> t=(1-q)/(1+q); t=(1-cx)^(1/c) -> x=(1-t^c)/c
    q, c = mp.mpf(q), mp.mpf(c)
    t = (1-q)/(1+q)
    return -mp.expm1(c*mp.log(t))/c

cases = [
 ("F1","cdf",1e-300,0.7727472780992932),("F1","cdf",1e-16,0.7727472780992932),
 ("F1","cdf",1e-12,1.0),("F1","cdf",1e-8,3.0),("F1","cdf",5e-8,1e6),
 ("F2","ppf",1e-300,0.7727472780992932),("F2","ppf",1e-16,1.0),("F2","ppf",1e-12,1.0),("F2","ppf",1e-8,3.0),
 ("F3","pdf",0.3333333333333333,3.0),("F3","pdf",1e-6,1e6),
 ("F4","cdf",0.01,1e-14),("F4","cdf",1.0,1e-14),("F4","cdf",1.0,1e-8),
 ("F5","ppf",0.01,1e-14),("F5","ppf",0.5,1e-14),("F5","ppf",0.5,1e-8),
 ("F6","pdf",10.0,1e-14),("F6","pdf",2.0,1e-14),("F6","pdf",1.0,1e-8),
]
fn = {"pdf":pdf,"cdf":cdf_quad,"ppf":ppf}
for fid, m, a, c in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(fn[m](a, c))
    mp.mp.dps = 200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref = vals[1]
    extra = ""
    if m == "ppf":
        # check cdf(x_ref) == q with closed-form cdf
        x = ref; cc = mp.mpf(c)
        t = mp.exp(mp.log1p(-cc*x)/cc)
        extra = " cdfchk=%s" % mp.nstr(((1-t)/(1+t)-mp.mpf(a))/mp.mpf(a), 3)
    s = getattr(D, m)(a, c)
    if np.isfinite(s) and ref != 0:
        ulp = float(abs(mp.mpf(float(s))-ref)/abs(ref))/2.220446049250313e-16
    else:
        ulp = float('inf')
    print(f"{fid} {m}({a!r}, {c!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.2g} agree30={agree}{extra}")
