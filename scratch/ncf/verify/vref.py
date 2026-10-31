# Independent verifier reference for ncf (standard form).
# pdf: closed form exp(-l/2) * f_F(x; n, m) * 1F1((n+m)/2; n/2; l*n*x/(2(n*x+m)))
# cdf/sf: Poisson mixture of mpmath regularized betainc with y = t/(1+t), yc = 1/(1+t) formed exactly.
import mpmath as mp

def _f(v): return mp.mpf(float(v)) if not isinstance(v, mp.mpf) else v

def pdf(x, n, m, l):
    x, n, m, l = map(_f, (x, n, m, l))
    a, b = n/2, m/2
    fF = mp.exp(a*mp.log(n/m) + (a-1)*mp.log(x) - (a+b)*mp.log(1 + n*x/m) - mp.log(mp.beta(a, b)))
    if l == 0: return fF
    z = l*n*x/(2*(n*x+m))
    return mp.exp(-l/2) * fF * mp.hyp1f1(a+b, a, z)

def _mix(x, n, m, l, upper):
    x, n, m, l = map(_f, (x, n, m, l))
    a, b = n/2, m/2
    t = n*x/m
    y = t/(1+t); yc = 1/(1+t)
    mu = l/2
    S = mp.mpf(0); j = 0
    tol = mp.mpf(10)**(-(mp.mp.dps+5))
    while True:
        w = mp.exp(-mu) * (mu**j / mp.factorial(j) if mu > 0 else (1 if j == 0 else 0))
        if upper:
            term = mp.betainc(a+j, b, y, 1, regularized=True) if t < 1 else mp.betainc(b, a+j, 0, yc, regularized=True)
        else:
            term = mp.betainc(a+j, b, 0, y, regularized=True) if t < 1 else 1 - mp.betainc(b, a+j, 0, yc, regularized=True)
        S += w*term
        if mu == 0 or (j > mu and w < tol*abs(S)) : break
        if j > 100000: raise RuntimeError
        j += 1
    return S

def cdf(x, n, m, l): return _mix(x, n, m, l, False)
def sf(x, n, m, l): return _mix(x, n, m, l, True)

def both(fn, *args):
    out = []
    for d in (100, 200):
        with mp.workdps(d):
            out.append(fn(*args))
    with mp.workdps(30):
        ok = mp.almosteq(out[0], out[1], rel_eps=mp.mpf(10)**-30, abs_eps=0) or out[0] == out[1]
    return out[1], ok
