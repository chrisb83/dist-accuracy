import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
import scipy.special as sc
from quadref import P_quad

def partial(a, x, N, dps=60):
    """N-term partial sum of the cephes igam_series:
       ans = sum_{n=0}^{N} prod_{i=1..n} x/(a+i);  P = ans*x^a e^-x/Gamma(a+1)"""
    mp.dps = dps
    a = mp.mpf(a); x = mp.mpf(x)
    r = a; c = mp.mpf(1); ans = mp.mpf(1)
    terms = []
    for n in range(1, N+1):
        r += 1
        c *= x/r
        ans += c
        terms.append(c/ans)
    ax = mp.e**(a*mp.log(x) - x - mp.loggamma(a))
    return ans*ax/a, terms

MACHEP = 1.11022302462515654042e-16
for a, x in [(1e6, 995000.0), (1e6, 990000.0), (1e8, 99954000.0), (1e9, 999841886.11699164)]:
    s = sc.gammainc(a, x)
    # how many terms until c/ans <= MACHEP ?
    mp.dps = 60
    aa = mp.mpf(a); xx = mp.mpf(x)
    r = aa; c = mp.mpf(1); ans = mp.mpf(1); nconv = None
    snap = {}
    for n in range(1, 400001):
        r += 1; c *= xx/r; ans += c
        if n in (1999, 2000, 2001, 2005, 2050):
            snap[n] = +ans
        if nconv is None and c/ans <= MACHEP:
            nconv = n
            if max(snap) >= 2050 or n < 2050:
                break
        if nconv is not None and n >= 2050:
            break
    ax = mp.e**(aa*mp.log(xx) - xx - mp.loggamma(aa))
    pref = ax/aa
    print("a=%g x=%.12g  nconv=%s  scipy=%.17g" % (a, x, nconv, s))
    for n in sorted(snap):
        v = snap[n]*pref
        print("    N=%4d partial=%s   rel diff to scipy=%.3g" % (n, mp.nstr(v, 18), float(abs(mp.mpf(s)-v)/v)))
