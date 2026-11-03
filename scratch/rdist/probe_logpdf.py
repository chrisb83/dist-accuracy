"""Decompose rdist._logpdf = -log2 + xlog1py(a-1,-y) + xlogy(a-1,y) - betaln(a,a) for large c."""
import mpmath as mp, numpy as np, scipy.special as sc
from scipy import stats
mp.mp.dps = 60
def lb(a, b): return mp.loggamma(a)+mp.loggamma(b)-mp.loggamma(a+b)
for c, x in [(1e3, 0.0), (1e3, 0.01), (1e6, 0.0), (1e6, 0.001), (1e6, 0.01), (1e4, 0.0)]:
    a = c/2; y = (x+1)/2
    t1 = sc.xlog1py(a-1, -y); t2 = sc.xlogy(a-1, y); bl = sc.betaln(a, a)
    T1 = (a-1)*mp.log1p(-mp.mpf(y)); T2 = (a-1)*mp.log(mp.mpf(y)); BL = lb(mp.mpf(a), mp.mpf(a))
    ref = (mp.mpf(c)/2-1)*mp.log(1-mp.mpf(x)**2) - lb(mp.mpf(0.5), mp.mpf(c)/2)
    got = stats.rdist.logpdf(x, c)
    ulp = lambda v, r: float(abs((mp.mpf(v)-r)/r))/2.22e-16
    print(f"c={c:g} x={x:g}: terms {float(T1):.6g} {float(T2):.6g} betaln {float(BL):.6g}; result {got!r} ref {mp.nstr(ref,17)} ulp {ulp(got,ref):.3g}")
    print(f"   abs err: xlog1py {float(t1-T1):.2e} xlogy {float(t2-T2):.2e} betaln {float(bl-BL):.2e} (betaln {ulp(bl,BL):.3g} ulp); total abs {float(mp.mpf(got)-ref):.2e}")
    # direct alternative: (c/2-1)*log1p(-x*x) - betaln(0.5, c/2)
    alt = (c/2-1)*np.log1p(-x*x) - sc.betaln(0.5, c/2)
    print(f"   alt (c/2-1)*log1p(-x*x) - betaln(0.5,c/2): {alt!r} ulp {ulp(alt,ref):.3g}")
