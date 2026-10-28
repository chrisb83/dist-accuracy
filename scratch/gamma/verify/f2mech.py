from mpmath import mp
import numpy as np, scipy.special as sc
from scipy.stats import gamma
mp.dps = 60
EPS = 2.220446049250313e-16
for x, a in [(1e6, 1e6), (1010000.0, 1e6), (1003000.0, 1e6), (999.0, 1000.0)]:
    aa = mp.mpf(a); xx = mp.mpf(x)
    lref = (aa-1)*mp.log(xx) - xx - mp.loggamma(aa)
    lsp = gamma.logpdf(x, a)
    pref = mp.e**lref
    psp = gamma.pdf(x, a)
    t1 = float(sc.xlogy(a-1.0, x)); t2 = x; t3 = float(sc.gammaln(a))
    print("x=%.8g a=%g" % (x, a))
    print("   terms: xlogy=%.6g  x=%.6g  gammaln=%.6g  -> logpdf=%.10g" % (t1, t2, t3, lsp))
    print("   logpdf abs err = %.3g   -> predicted pdf rel err = %.3g"
          % (float(abs(mp.mpf(lsp)-lref)), float(abs(mp.mpf(lsp)-lref))))
    print("   pdf rel err    = %.3g (%.3g ulp);  exp(logpdf_scipy) rel err = %.3g"
          % (float(abs(mp.mpf(psp)-pref)/pref), float(abs(mp.mpf(psp)-pref)/(pref*EPS)),
             float(abs(mp.e**mp.mpf(lsp)-pref)/pref)))
    print("   max term * eps = %.3g" % (max(abs(t1), abs(t2), abs(t3))*EPS))
