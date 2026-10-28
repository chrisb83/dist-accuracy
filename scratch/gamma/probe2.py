import numpy as np, mpmath as mp, scipy.special as sc
EPS = np.finfo(float).eps
def ulp(v, ref):
    ref = mp.mpf(ref)
    return float('nan') if ref == 0 else float(abs(mp.mpf(v)-ref)/(abs(ref)*EPS))
def P(a, x, dps=60):
    old = mp.mp.dps; mp.mp.dps = dps
    try:
        a = mp.mpf(a); x = mp.mpf(x)
        r = mp.gammainc(a, 0, x, regularized=True) if x < a else 1 - mp.gammainc(a, x, mp.inf, regularized=True)
    finally:
        mp.mp.dps = old
    return r

print("== cross-check of the reference at a=1e6, x=995000 ==")
a, x = 1e6, 995000.0
for dps in [30, 60, 150, 400]:
    print(f"  dps={dps:4d}  P = {mp.nstr(P(a,x,dps), 25)}")
mp.mp.dps = 60
# independent: P = x**a e**-x / Gamma(a+1) * hyp1f1(1, a+1, x)  (Kummer)
alt = mp.exp(a*mp.log(mp.mpf(x)) - mp.mpf(x) - mp.loggamma(a+1)) * mp.hyp1f1(1, a+1, mp.mpf(x))
print("  via hyp1f1:", mp.nstr(alt, 25))
# independent: integral of the pdf
mp.mp.dps = 40
quad = mp.quad(lambda t: mp.exp((a-1)*mp.log(t) - t - mp.loggamma(a)), [0, 990000, x])
print("  via quad  :", mp.nstr(quad, 20))
print("  scipy     :", repr(sc.gammainc(a, x)))

print("\n== a = 1e6: error vs k = (x-a)/sqrt(a) ==")
mp.mp.dps = 60
a = 1e6
for k in np.arange(-1, -21.5, -1.0):
    x = float(a + k*np.sqrt(a))
    r = P(a, x)
    print(f"  k={k:+6.1f} P={float(r):.4e} ulp={ulp(sc.gammainc(a,x), r):10.3g} rel={float(abs(mp.mpf(sc.gammainc(a,x))-r)/r):.2e}")

print("\n== fixed k = -5: error vs a ==")
for a in [100.0, 150.0, 199.0, 201.0, 300.0, 1e3, 1e4, 1e5, 3e5, 1e6]:
    x = float(a - 5*np.sqrt(a))
    r = P(a, x)
    print(f"  a={a:9.4g} x={x:.10g} P={float(r):.4e} ulp={ulp(sc.gammainc(a,x), r):10.3g}")
