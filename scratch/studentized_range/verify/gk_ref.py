# Verifier reference for general k: nested mp.quad (tanh-sinh), outer over u = log s.
# cdf(x) = int f_S(s) R_k(x s) ds,  R_k(w) = k int phi(z) D^(k-1) dz
# pdf(x) = int s f_S(s) r_k(x s) ds, r_k(w) = k(k-1) int phi(z) phi(z+w) D^(k-2) dz
# D = Phi(z+w) - Phi(z), computed as 0.5*(erfc(-(z+w)/sqrt2)-erfc(-z/sqrt2)) for z+w/2<0
# and 0.5*(erfc(z/sqrt2)-erfc((z+w)/sqrt2)) otherwise (no loss from 1-ish values).
import mpmath as mp
from k2_ref import logfS

def D(z, w):
  with mp.workdps(mp.mp.dps + 60):
    r2 = mp.sqrt(2)
    if z + w/2 < 0:
        r = (mp.erfc(-(z+w)/r2) - mp.erfc(-z/r2))/2
    else:
        r = (mp.erfc(z/r2) - mp.erfc((z+w)/r2))/2
  return +r

def zbreaks(w):
    c = -w/2
    return [mp.ninf, c-12, c-4, c-1, c, c+1, c+4, c+12, mp.inf]

def Rk(w, k):
    return k*mp.quad(lambda z: mp.npdf(z)*D(z, w)**(k-1), zbreaks(w))

def rk(w, k):
    return k*(k-1)*mp.quad(lambda z: mp.npdf(z)*mp.npdf(z+w)*D(z, w)**(k-2), zbreaks(w))

def ubreaks(x, df):
    w = 1/mp.sqrt(2*df)
    if df > 1000:
        pts = [k*w for k in range(-12, 13)]
        return [pts[0]-1] + pts + [pts[-1]+1]
    c = -mp.log(x)
    lo = min(c - 40, mp.mpf(-40)) if df >= 1 else min(c-60, mp.mpf(-300))
    return [lo, -20, -10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]

def cdf(x, k, df):
    x, k, df = mp.mpf(x), mp.mpf(k), mp.mpf(df)
    return mp.quad(lambda u: mp.exp(logfS(mp.exp(u), df) + u)*Rk(x*mp.exp(u), k), ubreaks(x, df))

def pdf(x, k, df):
    x, k, df = mp.mpf(x), mp.mpf(k), mp.mpf(df)
    return mp.quad(lambda u: mp.exp(logfS(mp.exp(u), df) + 2*u)*rk(x*mp.exp(u), k), ubreaks(x, df))
