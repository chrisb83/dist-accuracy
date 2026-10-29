import mpmath as mp, numpy as np
from scipy import stats
S1=(9.13259764654189, 16.23195660059063, 3.2819552690843983)
def ymexp(y):
    # y + expm1(-y) = sum_{k>=2} (-y)^k/k!, series for small y
    if abs(y) < mp.mpf('0.5'):
        return mp.nsum(lambda k: (-y)**int(k)/mp.factorial(int(k)), [2, mp.inf])
    return y + mp.expm1(-y)
def E(x,a,b,c): return -a*x - b/c*ymexp(c*x)
def pdf(x,a,b,c): return (a+b*(-mp.expm1(-c*x)))*mp.exp(E(x,a,b,c))
def logpdf(x,a,b,c): return mp.log(a)+mp.log1p(b*(-mp.expm1(-c*x))/a)+E(x,a,b,c)
def cdf_quad(x,a,b,c):  # integral of pdf, different route
    return x*mp.quad(lambda u: pdf(x*u,a,b,c), [0, 1])
def sf_quad(x,a,b,c):
    r = a + b*mp.exp(-c*x)   # local decay rate
    pts = [x + k/r for k in (0,1,2,4,8,16,32,64,128,256)] + [mp.inf]
    return mp.quad(lambda t: pdf(t,a,b,c), pts)
def ppf(p,a,b,c):
    # solve -E(x) = -log1p(-p) for x; Newton on log-scale: G(x)= -E(x)
    target = -mp.log1p(-p)
    x0 = target/a if target/a < 1/c else target/(a+b)
    lo, hi = mp.mpf(0), max(x0*4, mp.mpf(1))
    while -E(hi,a,b,c) < target: hi*=2
    # bisection in log space then findroot
    f=lambda x: -E(x,a,b,c)-target
    # use bisection to good bracket
    for _ in range(60):
        m=(lo+hi)/2 if lo>0 else hi/2**8 if hi>target*1e300 else (lo+hi)/2
        m=(lo+hi)/2
        if f(m)>0: hi=m
        else: lo=m
    lo = mp.mpf(0)
    # exact: use findroot with secant start
    x=mp.findroot(f, x0 if x0>0 else hi, solver='newton', df=lambda x: (a+b*(-mp.expm1(-c*x))))
    return x
def isf(q,a,b,c): return ppf(1-q,a,b,c)
import sys
cases=[
 ('F1','ppf',1e-20,(1,1,1)),('F1','ppf',1e-300,S1),('F1','ppf',1e-8,(1,2,0.001)),
 ('F1','ppf',0.01,(1000,10000,0.1)),('F1','ppf',0.01,(1,2,0.001)),('F1','ppf',0.9991516571017559,(1000,10000,0.1)),
 ('F2','isf',0.9999999999999997,(1e-6,10,2)),('F2','isf',0.9999999999999627,(1,2,0.001)),
 ('F2','isf',0.9999999999999997,(1000,10000,0.1)),('F2','isf',0.9999999999999966,S1),
 ('F2','isf',0.9907894468231052,(1000,10000,0.1)),('F2','isf',0.99,(1000,10000,0.1)),
 ('F2','isf',3.4807005884285624e-14,(1000,10000,0.1)),
 ('F3','logpdf',1e-10,(1,1,1)),('F3','logpdf',1e-150,(1,1,1)),('F3','logpdf',1e-8,(1,1,1)),
 ('F3','logpdf',1e-70,(1,2,0.001)),('F3','logpdf',1e-10,(1,1e-6,1)),
 ('F4','cdf',1e-190,(1e-6,10,2)),('F4','cdf',1e-8,(1e-6,10,2)),('F4','cdf',1e-8,(0.001,1,1)),
 ('F5','pdf',0.5,(1000,10000,0.1)),('F5','sf',0.5,(1000,10000,0.1)),
 ('F5','pdf',0.06772727272727273,(1000,10000,0.1)),('F5','pdf',253.33333333333334,(1,2,0.001)),
]
def ref(meth,x,s,dps):
    mp.mp.dps=dps
    x=mp.mpf(x); s=[mp.mpf(v) for v in s]
    if meth=='ppf': r=ppf(x,*s); chk=cdf_quad(r,*s)/x-1
    elif meth=='isf': r=isf(x,*s); chk=sf_quad(r,*s)/x-1
    elif meth=='logpdf': r=logpdf(x,*s); chk=r/mp.log(pdf(x,*s))-1 if x>1e-20 else 0
    elif meth=='cdf': r=-mp.expm1(E(x,*s)); chk=cdf_quad(x,*s)/r-1
    elif meth=='sf': r=mp.exp(E(x,*s)); chk=sf_quad(x,*s)/r-1
    elif meth=='pdf': r=pdf(x,*s); chk=0
    return r,chk
sel=sys.argv[1:]
for fid,meth,x,s in cases:
    if sel and fid not in sel: continue
    D=(300,400) if (meth=='logpdf' and x<1e-100) else (100,200)
    r1,c1=ref(meth,x,s,D[0]); r2,c2=ref(meth,x,s,D[1])
    mp.mp.dps=200
    agree = abs(r1-r2)/abs(r2) if r2!=0 else abs(r1)
    sp=getattr(stats.genexpon,meth)(x,*s)
    rel = abs(mp.mpf(float(sp))-r2)/abs(r2)
    ulp = rel/mp.mpf(2.220446049250313e-16)
    print(fid,flush=True,*[]) if False else print(fid,meth,repr(x),s,'| scipy',repr(float(sp)),'| ref',mp.nstr(r2,17),'| ulp',mp.nstr(ulp,3),'| agree',mp.nstr(agree,3),'| chk',mp.nstr(c2,3))
