import mpmath as mp, numpy as np, scipy, scipy.stats as st
cases = [
 ("F1","logpdf",1e10,1e10,10000000000.000002),
 ("F1","pdf",1e10,1e10,10000000000.000002),
 ("F1","logpdf",1.00000000005e300,1e300,1.0000000001e300),
 ("F1","pdf",1.00000000005e300,1e300,1.0000000001e300),
 ("F1","cdf",1.000000000099907e300,1e300,1.0000000001e300),
 ("F1","cdf",10000000500.0,1e10,1.000001e10),
 ("F1","cdf",10000009999.999994,1e10,1.000001e10),
 ("F1","pdf",10000005500.0,1e10,1.000001e10),
 ("F1","logpdf",10000005500.0,1e10,1.000001e10),
 ("F1","cdf",1.25e-300,1e-300,2e-300),
 ("F2","cdf",0.010000000000000002,0.01,1.25),
 ("F2","cdf",1.0000000000000002e300,1e300,1.0000000001e300),
 ("F2","cdf",1.0000000013815461e-300,1e-300,1e300),
 ("F2","cdf",1.0000000001e-300,1e-300,2e-300),
 ("F2","cdf",0.010000000482831391,0.01,1.25),
 ("F2","cdf",1.013911385736686e-300,1e-300,1e300),
]
def integ(lo, hi):
    # integral of 1/t from lo to hi, by quad on geometric subdivision (route != log formula)
    n = max(1, int(mp.ceil(mp.log10(hi/lo)/5)))
    r = (hi/lo)**(mp.mpf(1)/n)
    pts = [lo*r**k for k in range(n)] + [hi]
    return mp.quad(lambda t: 1/t, pts)
def ref(m, x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    if m in ("pdf","logpdf"):
        L = mp.log1p((b-a)/a)   # different identity from log(b/a)
        p = 1/(x*L)
        return p if m=="pdf" else mp.log(p)
    if m == "cdf":
        return integ(a, x)/integ(a, b)
def at(dps, *c):
    mp.mp.dps = dps; return ref(*c)
print("scipy", scipy.__version__)
for fid, m, x, a, b in cases:
    r1 = at(100, m, x, a, b); r2 = at(200, m, x, a, b)
    mp.mp.dps = 200
    agree = r1 == r2 or abs(r1-r2) <= abs(r2)*mp.mpf(10)**-30
    with np.errstate(all='ignore'):
        s = getattr(st.loguniform, m)(x, a, b)
    ulp = float(abs(mp.mpf(s)-r2)/abs(r2)/mp.mpf(2)**-52) if np.isfinite(s) else float('inf')
    print(f"{fid} {m}({x!r}, {a!r}, {b!r}) scipy={s!r} ref={mp.nstr(r2,17)} ulp={ulp:.3g} agree30={agree}")
