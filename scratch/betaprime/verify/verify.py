# Independent verifier reference for betaprime: pdf from docstring formula,
# cdf/sf via the 2F1 series representation of the incomplete beta, quantiles by findroot
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")

def B(a,b): return mp.beta(a,b)
def pdf(x,a,b): return x**(a-1)*(1+x)**(-a-b)/B(a,b)
def logpdf(x,a,b): return (a-1)*mp.log(x)-(a+b)*mp.log1p(x)-mp.log(B(a,b))
def _ib(y,yc,a,b):  # I_y(a,b) = y^a yc^b /(a B) 2F1(a+b,1;a+1;y)
    return y**a*yc**b/(a*B(a,b))*mp.hyp2f1(a+b,1,a+1,y)
def cdf(x,a,b):
    y=x/(1+x); yc=1/(1+x)
    return _ib(y,yc,a,b) if x<=1 else 1-_ib(yc,y,b,a)
def sf(x,a,b):
    y=x/(1+x); yc=1/(1+x)
    return _ib(yc,y,b,a) if x>1 else 1-_ib(y,yc,a,b)
def ppf(q,a,b,x0):
    q=mp.mpf(q)
    if q<0.4:
        f=lambda t: mp.log(cdf(mp.exp(t),a,b))-mp.log(q)
    else:
        f=lambda t: mp.log(sf(mp.exp(t),a,b))-mp.log(1-q)
    t=mp.findroot(f, mp.log(x0), tol=mp.mpf(10)**(-mp.mp.dps+10))
    x=mp.exp(t)
    res = cdf(x,a,b)/q-1 if q<0.4 else sf(x,a,b)/(1-q)-1
    return x, res

cases=[
 ("F1","ppf",1e-300,2,10),("F1","ppf",1e-120,2,10),("F1","ppf",1e-200,5,2),
 ("F1","ppf",1e-09,0.5,2),("F1","ppf",1e-101,2,10),
 ("F2","pdf",0.3345527195017296,1e5,3e5),("F2","pdf",0.3333333333333333,1e5,3e5),
 ("F2","pdf",1e200,1000,0.5),("F2","pdf",6000.0,1000,0.5),
 ("F3","logpdf",0.330907878145448,1e5,3e5),("F3","logpdf",0.3333333333333333,1e5,3e5),
 ("F3","logpdf",0.001,0.001,2),("F3","logpdf",200000.0,1000,0.5),
 ("F4","ppf",0.6546077528522537,1000,0.5),("F4","ppf",0.6399999999999999,1000,0.5),
 ("F4","ppf",0.008176429292048043,2,0.001),("F4","ppf",0.49999999999999994,1000,0.5),
]
for fid,m,arg,a,b in cases:
    s=getattr(stats.betaprime,m)(arg,a,b)
    vals=[]
    for dps in (100,200):
        mp.mp.dps=dps
        A,Bb,X=mp.mpf(a),mp.mpf(b),mp.mpf(arg)
        if m=="ppf":
            # leading-order start, independent of scipy
            if arg<0.4: x0=(X*A*B(A,Bb))**(1/A)
            else: x0=((1-X)*Bb*B(A,Bb))**(-1/Bb)
            if x0>1 and arg<0.4: x0=mp.mpf(1)
            try: v,res=ppf(X,A,Bb,x0)
            except Exception: v,res=ppf(X,A,Bb,mp.mpf(1))
        else:
            v=globals()[m](X,A,Bb); res=0
        vals.append((v,res))
    (v1,r1),(v2,r2)=vals
    agree=mp.nstr(abs(v1-v2)/abs(v2),3)
    mp.mp.dps=50
    ref=float(v2)
    ulp=abs(mp.mpf(s)-v2)/abs(v2)/mp.mpf(2.0**-52) if np.isfinite(s) else mp.inf
    print(f"{fid} {m}({arg!r},{a},{b}) scipy={s!r} ref={ref!r} ulp={mp.nstr(ulp,3)} 100v200={agree} invres={mp.nstr(r2,3)}")
