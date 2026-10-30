# Independent verifier reference: cdf/sf by mp.quad of the docstring pdf,
# quantiles by closed form inverse then checked with quad-cdf.
import mpmath as mp, numpy as np
from scipy import stats
D = stats.laplace_asymmetric

def pdf(t, k):
    c = 1/(k + 1/k)
    return c*mp.exp(-t*k) if t >= 0 else c*mp.exp(t/k)

def cdf_quad(x, k):
    # integrate the docstring pdf in scaled variables u = t/k (x<0) or u = t*k (x>=0)
    x = mp.mpf(x); k = mp.mpf(k); c = 1/(k + 1/k)
    if x < 0:
        return c*k*mp.quad(mp.exp, [-mp.inf, x/k])
    left = c*k*mp.quad(mp.exp, [-mp.inf, 0])
    return left + (c/k)*mp.quad(lambda u: mp.exp(-u), [0, x*k])

def sf_quad(x, k):
    x = mp.mpf(x); k = mp.mpf(k); c = 1/(k + 1/k)
    if x >= 0:
        return (c/k)*mp.quad(lambda u: mp.exp(-u), [x*k, mp.inf])
    right = (c/k)*mp.quad(lambda u: mp.exp(-u), [0, mp.inf])
    return right + c*k*mp.quad(mp.exp, [x/k, 0])

def ppf_cf(q, k):
    # inverse via log1p identity (avoids forming 1 - q)
    q = mp.mpf(q); k = mp.mpf(k); q0 = k**2/(1+k**2)
    if q >= q0:
        return -mp.log1p(k**2 - q*(1+k**2))/k
    return k*mp.log(q*(1+k**2)/k**2)

def isf_cf(q, k):
    q = mp.mpf(q); k = mp.mpf(k); p0 = 1/(1+k**2)
    if q <= p0:
        return -mp.log(q*(1+k**2))/k
    return k*mp.log1p(k**-2 - q*(1+k**-2))

cases = [
 ('F1','cdf',1e-300,1e-150),('F1','cdf',1e130,1e-150),('F1','cdf',1e140,1e-150),
 ('F1','cdf',3e-9,1e-8),('F1','cdf',0.0,1e-8),('F1','cdf',10.0,1e-8),('F1','cdf',0.0,1e-3),('F1','cdf',0.3,1e-3),
 ('F2','sf',-1e-300,1e150),('F2','sf',-1e130,1e150),('F2','sf',-1e140,1e150),('F2','sf',-1e-300,1e8),
 ('F2','sf',-10.0,1e8),('F2','sf',-1e-300,1000.0),('F2','sf',-0.3,1000.0),
 ('F3','ppf',2.872984833353539e-288,1e-150),('F3','ppf',1.2115276586286942e-26,1e-150),
 ('F3','ppf',1.3000000000000002e-16,1e-8),('F3','ppf',9.999990000020002e-07,1e-3),
 ('F3','ppf',1.0000999999999997e-12,1e-8),('F3','ppf',1.0000000000000002e-06,1e-3),
 ('F3','ppf',1.0000000099999999e-08,1e-8),('F3','ppf',1e-08,1e-150),('F3','ppf',0.200000008,0.5),
 ('F4','isf',1.2115276586286942e-26,1e150),('F4','isf',1.3e-16,1e8),('F4','isf',9.999990000020002e-07,1000.0),
 ('F4','isf',0.9999990000010001,1e-3),('F4','isf',3.4807005884285624e-14,1e150),
 ('F4','isf',1.0000000099999999e-08,1e8),('F4','isf',0.200000008,2.0),
]
fn = {'cdf':cdf_quad,'sf':sf_quad,'ppf':ppf_cf,'isf':isf_cf}
import warnings; warnings.simplefilter('ignore')
for fid, m, a, k in cases:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(fn[m](a, k))
    mp.mp.dps = 100
    agree = vals[0] == 0 or abs(vals[0]-vals[1])/abs(vals[1]) < mp.mpf(10)**-30
    ref = vals[1]
    chk = ''
    if m in ('ppf','isf'):
        mp.mp.dps = 60
        target = mp.mpf(a)
        c = cdf_quad(ref, k) if m=='ppf' else sf_quad(ref, k)
        chk = mp.nstr(abs(c-target)/target, 3)
    s = float(getattr(D, m)(a, k))
    r = float(ref)
    ulp = float(abs(mp.mpf(s) - ref)/(abs(ref)*mp.mpf(2)**-52)) if ref != 0 else float("nan")
    print(f"{fid} {m}({a!r}, {k!r}) scipy={s!r} ref={r!r} ulp={ulp:.3g} agree30={agree} cdfcheck={chk}")
