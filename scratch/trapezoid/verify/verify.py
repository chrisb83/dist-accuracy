import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter('ignore')
def pdf(x,c,d):
    u=2/(d-c+1)
    if x<c: return u*x/c
    if x<=d: return u
    return u*(1-x)/(1-d)
def cdf(x,c,d):
    x,c,d=mp.mpf(x),mp.mpf(c),mp.mpf(d)
    # integrate on [0,1] after substitution t = x*s (mp.quad is unreliable on tiny absolute intervals)
    pts=[0]+[p/x for p in (c,d) if p<x]+[1]
    # mp.quad's convergence test is absolute, so normalise the integrand to O(1)
    sc=max(pdf(x*p,c,d) for p in pts[1:]) if x<=d else pdf(c,c,d)
    return sc*x*mp.quad(lambda s: pdf(x*s,c,d)/sc, pts)
def ppf(q,c,d):
    # left branch only (all reproducers): solve cdf(x)=q via Newton on log, starting from analytical guess scaled
    q,c,d=mp.mpf(q),mp.mpf(c),mp.mpf(d)
    # bisection on log scale (cdf increasing), independent of the closed-form inverse
    lo,hi=mp.mpf(0),c
    for _ in range(int(mp.mp.prec*1.1)+1200):
        mid=(lo+hi)/2
        if cdf(mid,c,d)<q: lo=mid
        else: hi=mid
    x=(lo+hi)/2
    return x
def both(f,*a):
    out=[]
    for dps in (100,200):
        mp.mp.dps=dps; out.append(f(*a))
    mp.mp.dps=200
    assert abs(out[0]-out[1])<=abs(out[1])*mp.mpf(10)**-30, out
    return out[1]
def ulp(s,r):
    if r==0: return 0
    return float(abs(mp.mpf(s)-r)/abs(r)/mp.mpf(2.220446049250313e-16))
cases=[('cdf',1e-250,1e-200,0.5),('cdf',5e-201,1e-200,0.5),('cdf',9.99999999e-201,1e-200,0.5),
('cdf',1e-170,1e-100,0.5),('cdf',1e-160,1e-20,0.9),('cdf',1e-155,1e-20,0.9),
('ppf',1e-300,1e-200,0.5),('ppf',6.66666666e-201,1e-200,0.5),('ppf',1e-250,1e-100,0.5),('ppf',1e-300,1e-20,0.9)]
for m,a,c,d in cases:
    r=both(cdf if m=='cdf' else ppf,a,c,d)
    s=getattr(stats.trapezoid,m)(a,c,d)
    extra=''
    if m=='ppf':
        mp.mp.dps=100; extra=' check cdf(ref)/q-1=%s'%mp.nstr(cdf(r,c,d)/mp.mpf(a)-1,3)
    print(m,a,c,d,repr(float(s)),mp.nstr(r,17),'%.3g'%ulp(s,r),extra)
