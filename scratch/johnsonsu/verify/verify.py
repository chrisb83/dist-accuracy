import mpmath as mp
import numpy as np
from scipy import stats

def phi(z):
    return mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)

def L(x):
    # log(x + sqrt(x^2+1)), using odd symmetry to avoid cancellation for x<0
    if x < 0:
        return -L(-x)
    return mp.log1p(x + x*x/(mp.sqrt(x*x + 1) + 1))

def pdf(x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    return b/mp.sqrt(x*x+1)*phi(a + b*L(x))

def Phi(z):
    return mp.erfc(-z/mp.sqrt(2))/2

def zq_lower(q):
    q = mp.mpf(q)
    g = mp.sqrt(-2*mp.log(q)) * (-1 if q < 0.5 else 1)
    return mp.findroot(lambda z: mp.log(Phi(z)) - mp.log(q), g)

def ppf(q, a, b):
    z = zq_lower(q)
    return mp.sinh((z - mp.mpf(a))/mp.mpf(b))

def isf(q, a, b):
    # sf(x) = Phi(-(a+b asinh x)) = q  -> -(z) solves Phi(.)=q
    z = -zq_lower(q)
    return mp.sinh((z - mp.mpf(a))/mp.mpf(b))

def cdf_quad(x, a, b):
    # Phi(a) + integral_0^x pdf (different route)
    x = mp.mpf(x)
    return Phi(mp.mpf(a)) + mp.quad(lambda t: pdf(t, a, b), [0, x])

calls = [
 ("F1","pdf",-1e300,0.5,0.001),("F1","pdf",1.35e154,0.5,0.001),("F1","pdf",1e200,0.5,0.001),("F1","pdf",1e200,0,0.01),
 ("F2","pdf",4.3e-99,0,1e100),("F2","pdf",3.88e-99,0,1e100),("F2","pdf",3.84e-99,0,1e100),("F2","pdf",3.8e-99,0,1e100),
 ("F3","ppf",4.906713932054622e-198,-30,0.5),("F3","ppf",4.906718833861835e-198,-30,0.5),
 ("F3","isf",4.906713932054622e-198,30,0.5),("F3","isf",0.005318616581433098,2.554395574161155,2.2482281679651965),
 ("F4","isf",0.15,0.5,0.001),("F4","ppf",1e-08,0,0.01),
]
F = {"pdf":pdf,"ppf":ppf,"isf":isf}
for fid, m, x, a, b in calls:
    vals = []
    for d in (100, 200):
        mp.mp.dps = d
        vals.append(F[m](x, a, b))
    mp.mp.dps = 100
    agree = abs(vals[0]-vals[1]) <= abs(vals[1])*mp.mpf(10)**-30
    ref = vals[1]
    s = getattr(stats.johnsonsu, m)(x, a, b)
    err = abs(mp.mpf(float(s)) - ref)/abs(ref)/mp.mpf(2.220446049250313e-16) if s != 0 else mp.inf
    extra = ""
    if m in ("ppf","isf") and fid == "F3":
        mp.mp.dps = 60
        c = cdf_quad(ref, a, b)
        target = mp.mpf(x) if m == "ppf" else 1 - mp.mpf(x)
        if m == "isf":
            # compare sf: 1 - c
            sfv = Phi(-mp.mpf(a)) - mp.quad(lambda t: pdf(t, a, b), [0, ref])
            extra = " quadcheck rel=%s" % mp.nstr(abs(sfv - mp.mpf(x))/mp.mpf(x), 3)
        else:
            extra = " quadcheck rel=%s" % mp.nstr(abs(c - mp.mpf(x))/mp.mpf(x), 3)
    print(fid, m, repr(x), a, b, "scipy=%r" % float(s), "ref=%s" % mp.nstr(ref, 17), "ulp=%s" % mp.nstr(err, 3), "agree30=%s" % agree, extra)
