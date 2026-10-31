# Quantile check: Newton on log(cdf) (q<1/2) or log(sf) (q>1/2), cdf/sf by
# normalized mp.quad of the docstring pdf (vref.py). Start = report value; the
# independent part is that cdf(x_ref) = q to many digits.
import sys, mpmath as mp, numpy as np, scipy.stats as st

def pdf(x, b):
    return x*mp.exp(-(x*x+b*b)/2)*mp.besseli(0, x*b)
def cdf(x, b):
    c = pdf(x, b); return c*x*mp.quad(lambda u: pdf(u*x, b)/c, mp.linspace(0, 1, 5))
def sf(x, b):
    c = pdf(x, b); return c*mp.quad(lambda t: pdf(t, b)/c, [x, x+mp.mpf(1)/4, x+1, x+4, x+60])
def ppf(q, b, x0):
    q, b, x = mp.mpf(q), mp.mpf(b), mp.mpf(x0)
    lo = q < 0.5
    for it in range(40):
        if lo:
            F = cdf(x, b); step = (mp.log(F) - mp.log(q)) * F/(x*pdf(x, b))   # in log x
            x = x*mp.exp(-step); rel = abs(step)
        else:
            S = sf(x, b); step = (mp.log(S) - mp.log(1-q)) * S/pdf(x, b)
            x = x + step; rel = abs(step/x)
        if rel < mp.mpf(10)**(-mp.mp.dps+8): break
    chk = cdf(x, b)/q-1 if lo else sf(x, b)/(1-q)-1
    return x, chk
q, b, x0 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
r = []
for d in (100, 200):
    mp.mp.dps = d; r.append(ppf(q, b, x0))
mp.mp.dps = 200
ref = r[1][0]
s = st.rice.ppf(q, b)
ulp = float(abs(mp.mpf(float(s))-ref)/ref)/2.220446049250313e-16 if np.isfinite(s) else float('inf')
print(f"ppf({q!r}, {b!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} agree100v200={mp.nstr(abs(r[0][0]-ref)/ref,3)} chk100={mp.nstr(r[0][1],3)} chk200={mp.nstr(r[1][1],3)}")
