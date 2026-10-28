import warnings, time
import mpmath as mp, numpy as np
from scipy import stats
def M(v): return mp.mpf(float(v))
def logB(a,b): return mp.loggamma(a)+mp.loggamma(b)-mp.loggamma(a+b)
def pdf(x,a,b):
    return mp.exp((a-1)*mp.log(x)+(b-1)*mp.log1p(-x)-logB(a,b))
# (2, b): sf(x) = (1-x)^b (1 + b x)  (binomial identity), cdf = 1 - that
def sf2(x,b): return (1-x)**b*(1+b*x)
def sv(meth,*args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        return float(getattr(stats.beta,meth)(*args)), len(w)
out=[]
for dps in (100,200):
    mp.mp.dps=dps
    b=M(1e6); r={}
    # F6 ppf(0.78,2,1e6): solve sf2(x)=1-0.78 ; isf(0.22,2,1e6): sf2(x)=0.22
    for name,q,target in (('ppf',0.78,1-M(0.78)),('isf',0.22,M(0.22))):
        x0=M(sv(name,q,2,1e6)[0])
        x=mp.findroot(lambda x: mp.log(sf2(x,b))-mp.log(target), x0)
        # independent check: quad of pdf from 0 to x
        c=mp.quad(lambda t: pdf(t,2,b), [0,x/4,x/2,x])
        r[name]=(x, c)
    # F8 sf and cdf at (1e5,3e5) via quad of pdf over tail
    a,b3=M(1e5),M(3e5); s=mp.sqrt(a*b3/((a+b3)**2*(a+b3+1)))
    xs=M(0.25684652341066566); xc=M(0.24315347658933434)
    r['sf']=(mp.quad(lambda t: pdf(t,a,b3),[xs,xs+s/4,xs+s,xs+3*s,xs+10*s,xs+40*s]),)
    r['cdf']=(mp.quad(lambda t: pdf(t,a,b3),[xc-40*s,xc-10*s,xc-3*s,xc-s,xc-s/4,xc]),)
    out.append(r)
mp.mp.dps=200
calls={'ppf':('ppf',0.78,2,1e6),'isf':('isf',0.22,2,1e6),'sf':('sf',0.25684652341066566,1e5,3e5),'cdf':('cdf',0.24315347658933434,1e5,3e5)}
for k,c in calls.items():
    r1,r2=out[0][k],out[1][k]
    v,nw=sv(*c)
    ulp=abs(M(v)-r2[0])/abs(r2[0])/mp.mpf(2)**-52
    print(c, 'scipy',repr(v),'ref',mp.nstr(r2[0],17),'ulp',mp.nstr(ulp,3),'rel100v200',mp.nstr(abs(r1[0]-r2[0])/abs(r2[0]),3),'check',[mp.nstr(z,17) for z in r2[1:]],'warn',nw)
