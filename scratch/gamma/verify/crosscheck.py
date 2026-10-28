import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
from mpmath import mp
import scipy.special as sc
from quadref import P_quad

def P_series(a, x, dps=60, tol=None):
    """Fully converged Kummer series, independent of the quadrature route."""
    mp.dps = dps + 15
    aa = mp.mpf(a); xx = mp.mpf(x)
    tol = mp.mpf(10)**(-(dps+10))
    r = aa; c = mp.mpf(1); ans = mp.mpf(1); n = 0
    while True:
        n += 1
        r += 1; c *= xx/r; ans += c
        if c/ans <= tol or n > 2000000:
            break
    val = ans*mp.e**(aa*mp.log(xx) - xx - mp.loggamma(aa))/aa
    mp.dps = dps
    return +val, n

def P_quad2(a, x, dps=100, maxdegree=14):
    """quadrature with peak-adapted panels"""
    mp.dps = dps + 25
    aa = mp.mpf(a); xx = mp.mpf(x)
    sq = mp.sqrt(aa); lg = mp.loggamma(aa)
    k = (aa - xx)/sq
    def f(v):
        t = xx - sq*v
        if t <= 0: return mp.mpf(0)
        return mp.e**((aa-1)*mp.log(t) - t - lg)*sq
    w = 1/(k+1)
    pts = [mp.mpf(0)]
    p = w
    while p < 80:
        pts.append(p); p *= 2
    pts.append(mp.mpf(80))
    val = mp.quad(f, pts, maxdegree=maxdegree)
    mp.dps = dps
    return +val

EPS = 2.220446049250313e-16
cases = [(1e6, 995000.0), (1e6, 992000.0), (1e6, 990000.0), (1e6, 970000.0),
         (1e8, 99954000.0), (1e8, 99900000.0), (1e8, 99950000.0), (1e9, 999841886.11699164)]
for a, x in cases:
    ser, n = P_series(a, x, dps=40)
    q1 = P_quad(a, x, dps=100)
    q2 = P_quad2(a, x, dps=100)
    mp.dps = 40
    s = sc.gammainc(a, x)
    try:
        mg = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True)
        mgs = "%.3g" % float(abs(mg-ser)/ser)
    except Exception:
        mgs = "n/a"
    print("a=%g x=%.12g  nterm=%d" % (a, x, n))
    print("   series   = %s" % mp.nstr(ser, 25))
    print("   quad(old)= %s  rel to series %.3g" % (mp.nstr(q1, 25), float(abs(q1-ser)/ser)))
    print("   quad(new)= %s  rel to series %.3g" % (mp.nstr(q2, 25), float(abs(q2-ser)/ser)))
    print("   mp.gammainc rel to series: %s" % mgs)
    print("   scipy=%.17g  relerr vs series = %.3g  ulp=%.3g" % (s, float(abs(mp.mpf(s)-ser)/ser), float(abs(mp.mpf(s)-ser)/(ser*EPS))))
