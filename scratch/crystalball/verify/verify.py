# Independent verifier reference for crystalball: quadrature route.
# Power-law integrals in u = log t, t = (B - x)/(m/beta) = 1 + (-beta - x)*beta/m.
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter('ignore')

def M(v): return mp.mpf(float(v))

def tail_int(u0, m):
    # int_{u0}^inf exp((1-m) u) du, by quadrature
    k = m - 1
    pts = [u0] + [u0 + c/k for c in (1, 10, 100, 1000)] + [mp.inf]
    return mp.quad(lambda u: mp.exp(-k*(u - u0)), pts) * mp.exp(-k*u0)

def parts(b, m):
    e = mp.exp(-b**2/2)
    P = e*(m/b)*tail_int(0, m)                      # power-law mass
    G = mp.quad(lambda t: mp.exp(-t**2/2), [-b, 0, mp.inf])  # gaussian mass
    return P, G, 1/(P + G), e

def u_of(x, b, m):
    return mp.log(1 + (-b - x)*b/m)

def pdf(x, b, m):
    P, G, N, e = parts(b, m)
    if x > -b: return N*mp.exp(-x**2/2)
    return N*e*mp.exp(-m*u_of(x, b, m))

def logpdf(x, b, m):
    P, G, N, e = parts(b, m)
    if x > -b: return mp.log(N) - x**2/2
    return mp.log(N) - b**2/2 - m*u_of(x, b, m)

def cdf(x, b, m):
    P, G, N, e = parts(b, m)
    if x <= -b:
        return N*e*(m/b)*tail_int(u_of(x, b, m), m)
    return N*(P + mp.quad(lambda t: mp.exp(-t**2/2), [-b, x]))

def sf(x, b, m):
    P, G, N, e = parts(b, m)
    if x <= -b:
        # power-law piece between x and -beta, integrated directly (no 1 - cdf)
        u0 = u_of(x, b, m)
        k = m - 1
        pl = e*(m/b)*mp.quad(lambda u: mp.exp(-k*u), [0, u0]) if u0 > 0 else 0
        return N*(pl + G)
    return N*mp.quad(lambda t: mp.exp(-t**2/2), [x, mp.inf])

def ppf(q, b, m):
    P, G, N, e = parts(b, m)
    if q < N*P:
        # q = N e (m/b) exp((1-m)u)/(m-1)  ->  u
        u = mp.log(q*(m-1)/(N*e*(m/b)))/(1-m)
        return -b - (m/b)*mp.expm1(u)
    # gaussian: Phi(x) = Phi(-b) + (q/N - P)/sqrt(2pi); solve on sf for accuracy
    s = (1 - q)/N            # = int_x^inf exp(-t^2/2)
    x0 = mp.sqrt(2)*mp.erfinv(1 - 2*s/mp.sqrt(2*mp.pi))
    return x0

CALLS = [
 ('F1','pdf',-2,1,150),('F1','cdf',-2,1,150),('F1','sf',-2,1,150),('F1','pdf',-1,1,1000),
 ('F1','pdf',-1e140,1e-150,3),('F1','cdf',-1e140,1e-150,3),('F1','sf',-1e140,1e-150,3),('F1','cdf',-1e200,1e-150,3),
 ('F2','pdf',-140,1,140),('F2','pdf',-54.06977288832496,1,140),('F2','cdf',-54.06977288832496,1,140),
 ('F2','pdf',-2511863.4315095823,2,50),('F2','cdf',-2511863.4315095823,2,50),('F2','pdf',-15848908.924611142,2,50),
 ('F2','cdf',-140,1,140),
 ('F3','ppf',0.1,1,150),('F3','ppf',0.001,1e-150,3),('F3','ppf',1e-300,1,140),('F3','ppf',1e-300,2,50),
 ('F3','ppf',6.8129206905797616e-251,2,50),
 ('F4','sf',-1,1e-10,2),('F4','sf',-1e-09,1e-10,2),('F4','sf',-2,0.001,3),('F4','sf',-2,0.5,1.001),
 ('F5','logpdf',-1.5,1,1e6),('F5','logpdf',-1.000000001,1,1e6),('F5','logpdf',-1.5,1,1000),('F5','logpdf',-1.5,1,140),
 ('F6','logpdf',1.5e154,2,3),
 ('F7','ppf',0.9999999999995953,1,150),('F7','ppf',0.9999999999999966,1,150),
 ('F8','ppf',0.5,10,3),('F8','ppf',0.49999999999999994,10,3),
 ('F9','ppf',0.49999999999999994,0.5,1.001),
]
F = dict(pdf=pdf, cdf=cdf, sf=sf, logpdf=logpdf, ppf=ppf)

def ref(meth, a, b, m, dps):
    with mp.workdps(dps):
        return F[meth](M(a), M(b), M(m))

for fid, meth, a, b, m in CALLS:
    r1 = ref(meth, a, b, m, 100); r2 = ref(meth, a, b, m, 200)
    with mp.workdps(200):
        agree = abs(r1 - r2) <= abs(r2)*mp.mpf(10)**-30
        chk = ''
        if meth == 'ppf':
            bb, mm, x = M(b), M(m), r2
            q = M(a)
            if q > 0.5:
                chk = mp.nstr((sf(x, bb, mm) - (1 - q))/(1 - q), 3)
            else:
                chk = mp.nstr((cdf(x, bb, mm) - q)/q, 3)
    s = float(getattr(stats.crystalball, meth)(a, b, m))
    rf = float(r2)
    if rf == 0 or not np.isfinite(s): ulp = float('inf') if s != rf else 0
    else:
        with mp.workdps(50): ulp = float(abs(mp.mpf(s) - r2)/abs(r2)/mp.mpf(2)**-52)
    print(f"{fid} {meth}({a!r}, {b!r}, {m!r}) scipy={s!r} ref={rf!r} ulp={ulp:.3g} agree30={agree} {('inv_relerr='+chk) if chk else ''}")
