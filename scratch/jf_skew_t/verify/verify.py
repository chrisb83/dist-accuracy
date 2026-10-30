"""Independent verifier references for jf_skew_t (standard form).
cdf/sf: series identity I_y(a,b) = y^a (1-y)^b/(a B) 2F1(a+b,1;a+1;y), with y and 1-y
formed without cancellation; pdf: log form with 1+-t = (r+-x)/r rewritten exactly;
quantiles: Newton in the beta variable (log space) or, near the median, in x via quad."""
import sys, warnings
import mpmath as mp
import numpy as np
import scipy.special as sc
from scipy import stats

def logC(a, b):
    return (a+b-1)*mp.log(2) + mp.log(mp.beta(a, b)) + mp.log(a+b)/2

def ys(x, a, b):
    # returns (y, 1-y) with y = (1+t)/2, computed without cancellation
    r = mp.sqrt(a+b+x*x)
    if x < 0:
        y = (a+b)/(2*r*(r-x)); return y, (r-x)/(2*r)
    else:
        w = (a+b)/(2*r*(r+x)); return (r+x)/(2*r), w

def I(a, b, y):
    if y == 0: return mp.mpf(0)
    return y**a*(1-y)**b/(a*mp.beta(a, b))*mp.hyp2f1(a+b, 1, a+1, y)

def Ireg(a, b, y, w):  # I_y(a,b) given y and w=1-y
    if y <= 0.5: return I(a, b, y)
    return 1 - I(b, a, w)

def pdf(x, a, b):
    y, w = ys(x, a, b)          # 1+t = 2y, 1-t = 2w
    return mp.exp((a+mp.mpf(1)/2)*mp.log(2*y) + (b+mp.mpf(1)/2)*mp.log(2*w) - logC(a, b))

def cdf(x, a, b):
    y, w = ys(x, a, b); return Ireg(a, b, y, w)

def sf(x, a, b):
    y, w = ys(x, a, b); return Ireg(b, a, w, y)

def xfromu(u, v, a, b):
    return (u - v)*mp.sqrt(a+b)/(2*mp.sqrt(u*v))

def ppf(q, a, b):
    q = mp.mpf(q)
    if q < 0.5:
        A, B, target = a, b, q        # solve I_u(a,b) = q
    else:
        A, B, target = b, a, 1 - q    # solve I_v(b,a) = 1-q, v = 1-u
    lt = mp.log(target)
    # starting guess in log space
    s = (mp.log(target) + mp.log(A) + mp.log(mp.beta(A, B)))/A
    g = sc.betaincinv(float(A), float(B), float(target))
    if 0 < g < 0.5 and float(target) > 1e-300:
        s = mp.log(g)
    for _ in range(200):
        u = mp.exp(s)
        if u >= 1: u = mp.mpf(1)/2; s = mp.log(u)
        Iu = I(A, B, u) if u <= 0.5 else 1 - I(B, A, 1-u)
        f = mp.log(Iu) - lt
        dens = u**(A-1)*(1-u)**(B-1)/mp.beta(A, B)
        ds = f/(u*dens/Iu)
        s -= ds
        if abs(ds) < mp.mpf(10)**(-mp.mp.dps+5): break
    u = mp.exp(s)
    if q < 0.5: return xfromu(u, 1-u, a, b), u
    return xfromu(1-u, u, a, b), u

def ppf_median(q, a, b):
    # x-space Newton on cdf(x)-1/2 = int_0^x pdf = q - 1/2, quad of the pdf
    q = mp.mpf(q); d = q - mp.mpf(1)/2
    x = d/pdf(mp.mpf(0), a, b)
    for _ in range(50):
        F = mp.quad(lambda t: pdf(t, a, b), [0, x]) - d
        dx = F/pdf(x, a, b); x -= dx
        if abs(dx) < abs(x)*mp.mpf(10)**(-mp.mp.dps+5): break
    return x

CASES = [
 ('F1','cdf',1e200,8,4), ('F1','pdf',-1e300,0.001,0.001),
 ('F2','pdf',1e9,8,4), ('F2','pdf',34641.016151377546,8,4),
 ('F3','cdf',-4472135.954999579,0.001,0.001), ('F3','sf',34641.016151377546,8,4),
 ('F4','pdf',3,600,600),
 ('F5','pdf',0.5,32,32), ('F5','pdf',0.5,40,40),
 ('F6','ppf',0.99999,2,0.5), ('F6','ppf',0.9999999999,2,0.5),
 ('F7','ppf',1e-200,0.5,2), ('F7','ppf',1e-200,0.5,0.5),
 ('F8','ppf',1e-10,0.5,2),
 ('F9','ppfm',0.4999999999,600,600),
 ('F10','cdf',-1.0,1000,5),
 ('F11','pdf',1e-10,1000,5),
 ('F12','ppf',9.085175756517067e-139,600,600),
]

def ref(m, x, a, b):
    X, A, B = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    if m == 'pdf': return pdf(X, A, B)
    if m == 'cdf': return cdf(X, A, B)
    if m == 'sf': return sf(X, A, B)
    if m == 'ppf': return ppf(X, A, B)[0]
    if m == 'ppfm': return ppf_median(X, A, B)

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    out = {}
    for dps in (100, 200):
        mp.mp.dps = dps
        out[dps] = [ref(m, x, a, b) for (_, m, x, a, b) in CASES]
    for (fid, m, x, a, b), r1, r2 in zip(CASES, out[100], out[200]):
        mp.mp.dps = 200
        agree = (r1 == r2) or abs(r1-r2) <= abs(r2)*mp.mpf(10)**-30
        meth = 'ppf' if m == 'ppfm' else m
        if fid == 'F5':
            s = getattr(stats.jf_skew_t, meth)(x, np.int64(a), np.int64(b))
        else:
            s = getattr(stats.jf_skew_t, meth)(x, a, b)
        rf = float(r2)
        if rf == 0 or not np.isfinite(s):
            ulp = float('inf') if s != rf else 0.0
        else:
            ulp = float(abs(mp.mpf(float(s)) - r2)/abs(r2))/2.220446049250313e-16
        print(f"{fid}\t{meth}({x!r}, {a}, {b})\tscipy={s!r}\tref={mp.nstr(r2, 17)}\tulp={ulp:.3g}\tagree30={agree}")
    # quantile self-check: cdf(x_ref) == q
    mp.mp.dps = 100
    for (fid, m, x, a, b) in CASES:
        if m.startswith('ppf'):
            xr = ref(m, x, a, b)
            c = cdf(xr, mp.mpf(a), mp.mpf(b)); q = mp.mpf(x)
            print(fid, 'check rel |cdf(xref)-q|/min(q,1-q) =', mp.nstr(abs(c-q)/min(q, 1-q), 3))
