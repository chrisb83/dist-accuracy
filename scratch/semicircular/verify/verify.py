import mpmath as mp, numpy as np, scipy.special as sc
from scipy import stats
# pdf (docstring): f(x) = 2/pi sqrt(1-x^2), -1<=x<=1
def cdf_quad(x):
    x = mp.mpf(x)
    return mp.quad(lambda t: 2/mp.pi*mp.sqrt((1-t)*(1+t)), [-1, x])
def ppf_ref(q):
    # different route: solve x*sqrt(1-x^2)+asin(x) = pi*(q-1/2) by findroot, check with quad cdf
    q = mp.mpf(q); t = mp.pi*(q - mp.mpf(1)/2)
    g = lambda x: x*mp.sqrt(1-x*x) + mp.asin(x) - t
    x = mp.findroot(g, t/2, tol=mp.mpf(10)**(-mp.mp.dps+5))
    return x, cdf_quad(x) - q
def both(f, *a):
    out = []
    for d in (100, 200):
        mp.mp.dps = d; out.append(f(*a))
    return out
print("F1 cdf")
for x in [-0.999999999999, -0.9999999999999999, -0.99999999999, -0.9999999999, -0.99999999, -0.999999, -0.9999, -0.999]:
    r1, r2 = both(cdf_quad, x)
    mp.mp.dps = 100
    agree = abs(r1 - r2) <= abs(r2)*mp.mpf(10)**-30
    s = stats.semicircular.cdf(x); r = float(r2)
    ulp = abs(s - r2)/(abs(r2)*mp.mpf(2)**-52)
    print(f"{x!r} scipy={s!r} ref={r!r} ulp={mp.nstr(ulp,3)} agree30={agree}")
print("F2 ppf")
for q in [0.4999999999999999, 0.49999999999999994, 0.5000000000000001, 0.49999999999999, 0.499999999999, 0.4999999999, 0.50000000000001, 0.500000000001, 0.5000000001]:
    (x1, c1), (x2, c2) = both(ppf_ref, q)
    mp.mp.dps = 100
    agree = abs(x1 - x2) <= abs(x2)*mp.mpf(10)**-30
    s = stats.semicircular.ppf(q); r = float(x2)
    ulp = abs(s - x2)/(abs(x2)*mp.mpf(2)**-52)
    print(f"{q!r} scipy={s!r} ref={r!r} ulp={mp.nstr(ulp,3)} agree30={agree} cdfresid={mp.nstr(c2,3)}")
mp.mp.dps = 50
for q in [0.4999999999999999, 0.5000000000000001, 0.49999999999999]:
    p = sc.betaincinv(1.5, 1.5, q)
    pt = mp.findroot(lambda p: mp.betainc(1.5, 1.5, 0, p, regularized=True) - mp.mpf(q), 0.5)
    print("betaincinv", q, repr(p), mp.nstr(pt, 20), "err ulp(0.5):", mp.nstr((mp.mpf(p)-pt)/mp.mpf(2)**-53, 4))
