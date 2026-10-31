# Independent verifier reference for rice: cdf/sf by mp.quad of the docstring pdf,
# quantiles by root finding on log(cdf) or log(sf), pdf by besseli and asymptotic check.
import sys, mpmath as mp
import numpy as np, scipy, scipy.stats as st

def pdf(x, b):
    x, b = mp.mpf(x), mp.mpf(b)
    return x*mp.exp(-(x*x+b*b)/2)*mp.besseli(0, x*b)

# mp.quad uses an absolute tolerance, so integrands are normalized to O(1) first.
def cdf(x, b):
    x = mp.mpf(x)
    c = pdf(x, b)
    return c*x*mp.quad(lambda u: pdf(u*x, b)/c, mp.linspace(0, 1, 17))

def sf(x, b):
    x = mp.mpf(x)
    c = pdf(x, b)
    return c*mp.quad(lambda t: pdf(t, b)/c, [x, x+mp.mpf(1)/4, x+1, x+3, x+8, x+20, x+60])

def ppf(q, b):
    q = mp.mpf(q); b = mp.mpf(b)
    if q < mp.mpf('0.5'):
        lq = mp.log(q)
        x0 = mp.sqrt(2*q*mp.exp(b*b/2))  # small-x start
        if x0 > b/2:
            x0 = None
        g = lambda lx: mp.log(cdf(mp.exp(lx), b)) - lq
        if x0 is None:
            # bracket by bisection on x
            lo, hi = mp.mpf(0), b
            for _ in range(60):
                m = (lo+hi)/2
                if cdf(m, b) < q: lo = m
                else: hi = m
            x0 = (lo+hi)/2
        return mp.exp(mp.findroot(g, (mp.log(x0), mp.log(x0)+mp.mpf('1e-3')), solver='secant', tol=mp.mpf(10)**(-mp.mp.dps+10)))
    else:
        lp = mp.log(1-q)
        lo, hi = b, b+80
        for _ in range(60):
            m = (lo+hi)/2
            if sf(m, b) > 1-q: lo = m
            else: hi = m
        g = lambda x: mp.log(sf(x, b)) - lp
        x0 = (lo+hi)/2
        return mp.findroot(g, (x0, x0+mp.mpf('1e-6')), solver='secant', tol=mp.mpf(10)**(-mp.mp.dps+10))

FUN = dict(pdf=pdf, cdf=cdf, ppf=ppf)
CASES = [
 ('F1','cdf',1e-10,15),('F1','cdf',0.1,15),('F1','cdf',0.001,14.143),('F1','cdf',10.0,40),
 ('F1','cdf',0.36,15),('F1','cdf',16.75,40),
 ('F2','ppf',1e-300,15),('F2','ppf',1e-60,15),('F2','ppf',1e-300,40),('F2','ppf',1.333521432163324e-188,40),
 ('F3','pdf',2e154,2e154),('F3','pdf',1e155,1e155),('F3','pdf',1e154,1e154),
 ('F4','cdf',2.95e-81,0.7749725210111873),('F4','cdf',3e-81,3.0),('F4','cdf',3e-80,1e-05),
 ('F5','ppf',3.333e-162,0.7749725210111873),('F5','ppf',1.1006941712522576e-163,3.0),
 ('F6','ppf',0.9999999999999997,15),('F6','ppf',0.9999999999999997,40),('F6','ppf',0.9999999999522942,40),
]
if __name__ != "__main__": raise SystemExit
sel = sys.argv[1:]
print('scipy', scipy.__version__)
for fid, m, a, b in CASES:
    if sel and fid not in sel: continue
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append(FUN[m](a, b))
    mp.mp.dps = 200
    agree = mp.nstr(abs(vals[0]-vals[1])/abs(vals[1]), 3)
    ref = vals[1]
    with np.errstate(all='ignore'):
        s = getattr(st.rice, m)(a, b)
    if np.isfinite(s) and ref != 0:
        ulp = float(abs(mp.mpf(float(s))-ref)/abs(ref))/2.220446049250313e-16
    else:
        ulp = float('inf')
    extra = ''
    if m == 'ppf':
        mp.mp.dps = 100
        qa = mp.mpf(a)
        extra = ' chk=' + (mp.nstr(cdf(ref, b)/qa-1, 3) if qa < 0.5 else mp.nstr(sf(ref, b)/(1-qa)-1, 3))
    print(f"{fid} {m}({a!r}, {b!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} relagree={agree}{extra}", flush=True)
