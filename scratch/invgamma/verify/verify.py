# Independent verifier references for invgamma (standard form).
# pdf: f(x,a) = x**(-a-1) exp(-1/x) / Gamma(a)  (docstring)
# sf(x) = int_x^inf f = P(a, 1/x)  computed via explicit Kummer series
#   P(a,y) = y^a e^-y / Gamma(a+1) * sum_k y^k/((a+1)_k)
# cdf small-tail Q(a,y) via mp.quad of the gamma density on [y, inf)
# quantiles: findroot on the series / quad, then check cdf(x_ref)==q
import mpmath as mp
import numpy as np
from scipy import stats

def pdf(x, a):
    x = mp.mpf(x); a = mp.mpf(a)
    return x**(-a-1) * mp.exp(-1/x) * mp.rgamma(a)

def logpdf(x, a):
    return mp.log(pdf(x, a))

def P(a, y):
    a = mp.mpf(a); y = mp.mpf(y)
    s = mp.mpf(1); t = mp.mpf(1); k = 0
    while True:
        k += 1
        t *= y / (a + k)
        s += t
        if t < s * mp.eps * mp.mpf(10)**-5 and y/(a+k) < 0.99:
            break
    return mp.exp(a*mp.log(y) - y - mp.loggamma(a+1)) * s

def Qquad(a, y):
    a = mp.mpf(a); y = mp.mpf(y)
    g = lambda t: mp.exp((a-1)*mp.log(t) - t - mp.loggamma(a))
    pts = [y, y+1, y+10, y+50, y+200, y+1000, y+5000, mp.inf]
    return mp.quad(g, pts)

def sf(x, a):  return P(a, 1/mp.mpf(x))
def cdf_viaP(x, a): return 1 - P(a, 1/mp.mpf(x))
def cdf_quad(x, a): return Qquad(a, 1/mp.mpf(x))

def solve(fun, target, x0):
    # solve fun(y) = target in log y
    tgt = mp.log(mp.mpf(target))
    h = lambda u: mp.log(fun(mp.exp(u))) - tgt
    u = mp.findroot(h, mp.log(mp.mpf(x0)), tol=mp.mpf(10)**(-mp.mp.dps+10))
    return mp.exp(u)

def ppf(q, a, x0):   # cdf(x)=q  <=>  P(a,1/x) = 1-q
    y = solve(lambda y: P(a, y), 1 - mp.mpf(q), 1/mp.mpf(x0))
    return 1/y
def isf(q, a, x0):   # sf(x)=q <=> P(a,1/x) = q
    y = solve(lambda y: P(a, y), mp.mpf(q), 1/mp.mpf(x0))
    return 1/y

cases = [
 ('F1','sf',1.0050251256281407e-06,1e6),
 ('F1','sf',1.0080645161290323e-06,1e6),
 ('F1','cdf',1.0050251256281407e-06,1e6),
 ('F2','pdf',1.0050251256281407e-06,1e6),
 ('F2','pdf',1e-06,1e6),
 ('F2','pdf',1e-08,1e8),
 ('F3','logpdf',1.0050251256281407e-06,1e6),
 ('F3','logpdf',1e-08,1e8),
 ('F4','ppf',0.9999993371296838,1e6),
 ('F4','ppf',0.999999938945977,1e6),
 ('F4','isf',1e-08,1e6),
 ('F5','ppf',0.4999999999999999,1e-3),
 ('F5','isf',0.8499999999999999,1e-3),
 ('F6','cdf',0.0005,1000),
 ('F6','sf',0.002,1000),
]

def ref(m, arg, a, sp):
    if m == 'pdf': return pdf(arg, a)
    if m == 'logpdf': return logpdf(arg, a)
    if m == 'sf': return sf(arg, a)
    if m == 'cdf':
        return cdf_quad(arg, a) if a == 1000 else cdf_viaP(arg, a)
    if m == 'ppf': return ppf(arg, a, sp)
    if m == 'isf': return isf(arg, a, sp)

if __name__ == "__main__":
  for fid, m, arg, a in cases:
      sp = float(getattr(stats.invgamma, m)(arg, a))
      vals = []
      for d in (100, 200):
          mp.mp.dps = d
          vals.append(ref(m, arg, a, sp))
      mp.mp.dps = 200
      agree = abs(vals[0]-vals[1]) / abs(vals[1])
      r = vals[1]
      chk = ''
      if m == 'ppf': chk = ' cdfcheck=%s' % mp.nstr(cdf_viaP(r, a)/ (mp.mpf(arg)) - 1 if False else (1-P(a,1/r)) - mp.mpf(arg), 5)
      if m == 'isf': chk = ' sfcheck=%s' % mp.nstr(P(a,1/r)/mp.mpf(arg) - 1, 5)
      ulp = float(abs(mp.mpf(sp) - r) / abs(r)) / 2.220446049250313e-16
      print(f"{fid} {m}({arg!r}, {a:g}) scipy={sp!r} ref={mp.nstr(r,17)} ulp={ulp:.3g} dps_agree={float(agree):.1e}{chk}")
