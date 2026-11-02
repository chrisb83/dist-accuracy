import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
D = stats.truncexpon

def I(lo, hi):  # integral of exp(-t) over [lo, hi], rescaled to [0, 1] (raw quad is inaccurate on tiny intervals)
    h = hi - lo
    return h*mp.quad(lambda u: mp.exp(-(lo + h*u)), [0, 1])
def pdf(x, b): return mp.exp(-x)/I(0, b)
def sf_q(x, b):
    return I(x, b)/I(0, b)
def cdf_q(x, b):
    return I(0, x)/I(0, b)
def logpdf(x, b): return mp.log(pdf(x, b))
def isf(q, b):
    # closed-form alternative, checked by quadrature below
    return b - mp.log1p(q*mp.expm1(b))

cases = [
 ("F1","sf",(5e-301,1e-300)),("F1","sf",(5e-17,1e-16)),("F1","sf",(9.99999e-13,1e-12)),
 ("F1","sf",(5e-13,1e-12)),("F1","sf",(1e-300,1e-12)),("F1","sf",(9.5e-07,1e-06)),("F1","sf",(1e-300,1e-06)),
 ("F2","isf",(0.5,1e-300)),("F2","isf",(1e-300,1e-300)),("F2","isf",(0.9999999999999627,1e-06)),
 ("F2","isf",(0.5,1e-12)),("F2","isf",(1e-10,1e-12)),("F2","isf",(0.9999999,1e-6)),("F2","isf",(0.5,1e-6)),
 ("F3","logpdf",(0.,50.)),("F3","logpdf",(0.,38.)),("F3","logpdf",(1e-20,50.)),("F3","logpdf",(0.,36.)),
 ("F3","logpdf",(2e-9,20.)),("F3","logpdf",(0.,20.)),("F3","logpdf",(1e-12,50.)),
 ("F4","cdf",(1e-312,1e-12)),("F4","cdf",(1e-310,1e-300)),
]
fn = {"sf":sf_q,"cdf":cdf_q,"logpdf":logpdf,"isf":isf}
for fid, m, (a, b) in cases:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(fn[m](mp.mpf(a), mp.mpf(b)))
    mp.mp.dps = 200
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30 or vals[1] == 0
    ref = vals[1]
    extra = ""
    if m == "isf":
        chk = sf_q(ref, mp.mpf(b))
        extra = " sfcheck_relerr=%s" % mp.nstr(abs(chk-mp.mpf(a))/mp.mpf(a), 3)
    s = float(getattr(D, m)(a, b))
    err = abs(mp.mpf(s)-ref)/abs(ref) if ref != 0 else mp.inf
    ulp = err/mp.mpf(2.220446049250313e-16)
    print(f"{fid} {m}{(a,b)} scipy={s!r} ref={mp.nstr(ref,17)} ulp={mp.nstr(ulp,3)} agree30={agree}{extra}")
