import mpmath as mp, numpy as np, scipy, warnings
from scipy import stats
warnings.simplefilter("ignore")
def pdf(x): return mp.sqrt(2/mp.pi)*mp.exp(-x*x/2)
def ppf(q):  # route: erfinv identity, checked by quadrature of pdf
    return mp.sqrt(2)*mp.erfinv(q)
def logpdf(x): return mp.log(2/mp.pi)/2 - x*x/2
def ref(f, a, dps):
    with mp.workdps(dps): return f(mp.mpf(a))
cases = [("ppf",q) for q in [1e-300,1e-20,1e-16,1e-15,1e-12,1e-8,1e-6,1e-4,1-2**-53,1-2**-52]] + \
        [("logpdf",x) for x in [1.35e154,1.5e154,1.8e154,1.89e154]]
print("scipy", scipy.__version__)
for m,a in cases:
    r1=ref(globals()[m],a,100); r2=ref(globals()[m],a,200)
    with mp.workdps(200):
        agree = mp.almosteq(r1,r2,rel_eps=mp.mpf(10)**-30)
        chk=""
        if m=="ppf":
            q=mp.mpf(a); x=r2
            if q<0.5: c=mp.quad(pdf,[0,x]); chk=float(abs(c/q-1))
            else: c=mp.quad(pdf,[x,mp.inf]); chk=float(abs(c/(1-q)-1))
    s=getattr(stats.halfnorm,m)(a)
    rf=float(r2)
    ulp = abs(s-rf)/(abs(rf)*2.220446049250313e-16) if np.isfinite(s) else np.inf
    print(f"{m}({a!r}) scipy={s!r} ref={rf!r} ulp={ulp:.2g} agree30={agree} quadrelerr={chk}")
