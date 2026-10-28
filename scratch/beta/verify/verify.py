import sys, warnings
import mpmath as mp
import numpy as np
from scipy import stats

def M(v): return mp.mpf(float(v))

def logB(a,b): return mp.loggamma(a)+mp.loggamma(b)-mp.loggamma(a+b)
def logpdf(x,a,b):
    x,a,b=M(x),M(a),M(b)
    return (a-1)*mp.log(x)+(b-1)*mp.log1p(-x)-logB(a,b)
def pdf(x,a,b): return mp.exp(logpdf(x,a,b))
def cdf_hyp(x,a,b):
    # I_x(a,b) = x^a (1-x)^b / (a B) * 2F1(a+b,1;a+1;x)
    x,a,b=M(x),M(a),M(b)
    return mp.exp(a*mp.log(x)+b*mp.log1p(-x)-logB(a,b))/a*mp.hyp2f1(a+b,1,a+1,x)
def cdf_bi(x,a,b):
    return mp.betainc(M(a),M(b),0,M(x),regularized=True)
def sf_hyp(x,a,b):  # I_{1-x}(b,a)
    x=M(x); return cdf_hyp_y(1-x,b,a)
def cdf_hyp_y(y,a,b):
    a,b=M(a),M(b)
    return mp.exp(a*mp.log(y)+b*mp.log1p(-y)-logB(a,b))/a*mp.hyp2f1(a+b,1,a+1,y)
def logI(y,a,b):  # log I_y(a,b) via hyp series, y mp
    a,b=M(a),M(b)
    return a*mp.log(y)+b*mp.log1p(-y)-logB(a,b)-mp.log(a)+mp.log(mp.hyp2f1(a+b,1,a+1,y))
def small_root(q,a,b,y0=None):
    # solve I_y(a,b)=q for y, in t=log y
    q=M(q); lq=mp.log(q); a_,b_=M(a),M(b)
    if y0 is None:
        y0=mp.exp((lq+mp.log(a_)+logB(a_,b_))/a_)
    t=mp.findroot(lambda t: logI(mp.exp(t),a,b)-lq, mp.log(y0))
    return mp.exp(t)
def ppf(q,a,b,x0=None):
    q=M(q)
    if q<0.5: return small_root(q,a,b,x0)
    # upper: 1-x small, I_{1-x}(b,a)=1-q
    return 1-small_root(1-q,b,a, None if x0 is None else 1-M(x0))
def isf(q,a,b,x0=None):
    q=M(q)
    if q<0.5: y=small_root(q,b,a, None if x0 is None else 1-M(x0)); return 1-y, y
    x=small_root(1-q,a,b,x0); return x, 1-x

cases = [
 ('F1','ppf',1e-16,0.5,3),('F1','ppf',1e-200,5,2),('F1','ppf',1e-300,2,1e6),
 ('F1','ppf',1e-120,2,10),('F1','ppf',1e-9,0.5,2),
 ('F2','pdf',1e-310,0.5,5),('F2','pdf',5e-324,0.5,0.5),('F2','pdf',1e-310,0.001,0.001),('F2','pdf',1e-308,1.5,5),
 ('F3','isf',1e-300,1e6,2),('F3','isf',1e-200,1e6,2),('F3','isf',1e-200,5,2),
 ('F4','logpdf',0.9999838579178736,1e6,2),('F4','logpdf',0.24863069531786686,1e5,3e5),('F4','logpdf',0.25,1e5,3e5),
 ('F5','isf',1e-300,20,50),('F5','isf',1e-299,20,50),
 ('F6','ppf',0.78,2,1e6),('F6','isf',0.22,2,1e6),
 ('F8','sf',0.25684652341066566,1e5,3e5),('F8','pdf',0.25684652341066566,1e5,3e5),
 ('F8','cdf',0.24315347658933434,1e5,3e5),('F8','pdf',0.27644908299505205,1000,1000),
]

def ref(meth,x,a,b,sv):
    x0 = sv if (isinstance(sv,float) and np.isfinite(sv) and 0<sv<1 and meth in('ppf','isf') and 1e-3<x<0.999) else None
    if meth=='pdf': return pdf(x,a,b), None
    if meth=='logpdf': return logpdf(x,a,b), None
    if meth=='cdf': return cdf_hyp(x,a,b), cdf_bi(x,a,b)
    if meth=='sf': return sf_hyp(x,a,b), mp.betainc(M(a),M(b),M(x),1,regularized=True)
    if meth=='ppf':
        r=ppf(x,a,b,x0); return r, (cdf_bi(r,a,b) if r<0.5 else 1-mp.betainc(M(b),M(a),0,1-r,regularized=True))
    if meth=='isf':
        r,y=isf(x,a,b,x0); return r, (mp.betainc(M(b),M(a),0,y,regularized=True) if M(x)<0.5 else 1-cdf_bi(r,a,b)), y

def scipy_val(meth,x,a,b):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: v=float(getattr(stats.beta,meth)(x,a,b))
        except Exception as e: return type(e).__name__, w
    return v, w

import time
sel=[int(v) for v in sys.argv[1:]] or range(len(cases))
for k in sel:
    c=cases[k]; f,meth,x,a,b=c
    out=[]
    t0=time.time()
    for dps in (100,200):
        mp.mp.dps=dps
        sv,w=scipy_val(meth,x,a,b)
        out.append(ref(meth,x,a,b,sv))
    r1,r2=out
    mp.mp.dps=200
    agree = mp.nstr(abs(r1[0]-r2[0])/abs(r2[0]),3) if r2[0]!=0 else '0'
    refv=r2[0]
    if isinstance(sv,float):
        ulp='nan' if np.isnan(sv) else mp.nstr(abs(M(sv)-refv)/abs(refv)/mp.mpf(2)**-52,3)
    else: ulp='exc'
    extra=' '.join(mp.nstr(v,17) for v in r2[1:] if v is not None)
    print(f"[{k}] {f} {meth}({x!r}, {a}, {b}) scipy={sv!r} ref={mp.nstr(refv,17)} ulp={ulp} rel100v200={agree} check={extra} warn={[str(x.message)[:60] for x in w]} t={time.time()-t0:.1f}", flush=True)
