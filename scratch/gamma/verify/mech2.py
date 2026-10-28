import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
import scipy.special as sc
from quadref import P_quad
MACHEP = 1.11022302462515654042e-16
for a, x in [(1e6, 970000.0), (1e6, 985000.0), (1e6, 980000.0)]:
    mp.dps = 60
    aa = mp.mpf(a); xx = mp.mpf(x)
    r = aa; c = mp.mpf(1); ans = mp.mpf(1); nconv=None; snap={}
    for n in range(1, 200001):
        r += 1; c *= xx/r; ans += c
        if n == 2000: snap[2000] = +ans
        if c/ans <= MACHEP:
            nconv = n; break
    pref = mp.e**(aa*mp.log(xx) - xx - mp.loggamma(aa))/aa
    conv = ans*pref
    s = sc.gammainc(a, x)
    ref = P_quad(a, x, dps=80); mp.dps = 60
    print("a=%g x=%g nconv=%d scipy=%.17g" % (a, x, nconv, s))
    print("   converged series = %s  (rel to quad ref %.2g)" % (mp.nstr(conv,18), float(abs(conv-ref)/ref)))
    if 2000 in snap:
        p2000 = snap[2000]*pref
        print("   N=2000 partial    = %s  rel to scipy %.3g" % (mp.nstr(p2000,18), float(abs(mp.mpf(s)-p2000)/p2000)))
    print("   scipy rel err vs quad ref = %.3g" % float(abs(mp.mpf(s)-ref)/ref))
