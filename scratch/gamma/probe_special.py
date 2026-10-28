"""Direct accuracy probe of the scipy.special functions gamma_gen forwards to."""
import numpy as np, mpmath as mp
import scipy.special as sc
mp.mp.dps = 60
EPS = np.finfo(float).eps

def ulp(v, ref):
    ref = mp.mpf(ref)
    if ref == 0: return float('nan')
    return float(abs(mp.mpf(v) - ref) / (abs(ref) * EPS))

def P(a, x):   # lower regularized
    a = mp.mpf(a); x = mp.mpf(x)
    if x >= a: return 1 - mp.gammainc(a, x, mp.inf, regularized=True)
    return mp.gammainc(a, 0, x, regularized=True)

def Q(a, x):
    a = mp.mpf(a); x = mp.mpf(x)
    if x >= a: return mp.gammainc(a, x, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, x, regularized=True)

print("== gammainc / gammaincc for a = 1e6, x = a + k*sqrt(a) ==")
a = 1e6
for k in [-30, -20, -15, -12, -10, -8, -5, -2, 0, 2, 5, 8, 10, 12, 15, 20, 30]:
    x = a + k*np.sqrt(a)
    rp, rq = P(a, x), Q(a, x)
    print(f"k={k:+5.0f} x={x:.10g}  P={float(rp):.6e} ulp(gammainc)={ulp(sc.gammainc(a,x),rp):10.3g}"
          f"   Q={float(rq):.6e} ulp(gammaincc)={ulp(sc.gammaincc(a,x),rq):10.3g}")

print("\n== gammainc for a = 1e3, 1e4, 1e5, 1e7 at k = -10 ==")
for a in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
    x = a - 10*np.sqrt(a)
    rp = P(a, x)
    print(f"a={a:9.3g} x={x:.10g} P={float(rp):.4e} ulp={ulp(sc.gammainc(a,x),rp):10.3g}")

print("\n== gammaincinv / gammainccinv, a = 1e6 ==")
a = 1e6
for q in [1e-16, 1e-10, 1e-8, 1e-4, 0.01, 0.3, 0.5, 0.9]:
    v = sc.gammaincinv(a, q)
    ref = mp.findroot(lambda x: P(a, x) - mp.mpf(q), mp.mpf(v))
    print(f"gammaincinv(a,{q:g}) = {v!r:24} ref={mp.nstr(ref,17)} ulp={ulp(v,ref):10.3g}")
for q in [1e-16, 1e-8, 0.01, 0.5, 0.9, 1-1e-6, 1-6.6e-7, 1-1e-9]:
    v = sc.gammainccinv(a, q)
    ref = mp.findroot(lambda x: Q(a, x) - mp.mpf(q), mp.mpf(v))
    print(f"gammainccinv(a,{q:.12g}) = {v!r:24} ref={mp.nstr(ref,17)} ulp={ulp(v,ref):10.3g}")

print("\n== gammaincinv for small a (bulk) ==")
for a in [1e-3, 1e-2, 0.1]:
    for q in [0.36, 0.5, 0.85]:
        v = sc.gammaincinv(a, q)
        if not np.isfinite(v) or v == 0:
            print(f"a={a} q={q} -> {v}"); continue
        ref = mp.findroot(lambda x: mp.log(P(a, x)) - mp.log(mp.mpf(q)), mp.mpf(v))
        print(f"gammaincinv({a},{q}) = {v!r:24} ref={mp.nstr(ref,17)} ulp={ulp(v,ref):10.3g}  1/a={1/a:g}")

print("\n== cdf far left tail: exp(a*log x) amplification ==")
for a in [1.0, 1.9932305483800778]:
    for x in [1e-280, 1e-130, 1e-30]:
        rp = P(a, x)
        print(f"a={a} x={x:g} P={float(rp):.6e} ulp={ulp(sc.gammainc(a,x),rp):8.3g}  |a*log x|*eps/eps={abs(a*np.log(x)):.0f} ulp expected")
