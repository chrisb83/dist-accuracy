# Independent reference: factor out exp(-a^c); use expm1/log1p forms in mpmath,
# cross-check cdf by mp.quad of the pdf, quantiles by cdf(x_ref) == q.
import numpy as np, mpmath as mp, warnings
from scipy import stats
warnings.simplefilter('ignore')

def M(v): return mp.mpf(float(v))

def Z(c,a,b):  # normaliser times exp(a^c): 1 - exp(a^c - b^c)
    return -mp.expm1(a**c - b**c)
def pdf(x,c,a,b): return c*x**(c-1)*mp.exp(a**c - x**c)/Z(c,a,b)
def logpdf(x,c,a,b): return mp.log(c)+(c-1)*mp.log(x)+(a**c-x**c)-mp.log(Z(c,a,b))
def cdf(x,c,a,b): return -mp.expm1(a**c - x**c)/Z(c,a,b)
def sf(x,c,a,b): return mp.exp(a**c-x**c)*(-mp.expm1(x**c-b**c))/Z(c,a,b)
def logcdf(x,c,a,b): return mp.log(cdf(x,c,a,b))
def logsf(x,c,a,b): return mp.log(sf(x,c,a,b))
def ppf(q,c,a,b): return (a**c - mp.log1p(-q*Z(c,a,b)))**(1/c)
def isf(q,c,a,b): return ppf(1-q,c,a,b)
F = dict(pdf=pdf,logpdf=logpdf,cdf=cdf,sf=sf,logcdf=logcdf,logsf=logsf,ppf=ppf,isf=isf)

rows = [
 ('F1','pdf',30.5,(2,30,31)),('F1','logpdf',30,(2,30,31)),('F1','cdf',30.5,(2,30,31)),
 ('F1','logsf',30.5,(2,30,31)),('F1','ppf',0.5,(2,30,31)),('F1','isf',0.5,(2,30,31)),
 ('F1','pdf',27.29,(2,26.8,27.5)),('F1','ppf',0.9999999999995953,(2,26.8,27.5)),
 ('F1','cdf',26.8007,(2,26.8,27.5)),
 ('F2','pdf',5e-9,(2,0,7e-9)),('F2','logpdf',5e-9,(2,0,7e-9)),('F2','cdf',5e-9,(2,0,7e-9)),
 ('F2','sf',5e-9,(2,0,7e-9)),('F2','ppf',0.5,(2,0,7e-9)),('F2','pdf',5e-9,(2,0,1e-8)),
 ('F2','pdf',5e-7,(2,0,1e-6)),('F2','logpdf',1e-12,(2,0,1e-6)),('F2','ppf',0.5,(2,0,1e-6)),
 ('F2','isf',0.5,(2,0,1e-6)),('F2','pdf',1.000000005,(2,1,1.00000001)),
 ('F2','logpdf',1.000000005,(2,1,1.00000001)),('F2','pdf',0.005,(3,0,0.01)),
 ('F2','logpdf',0.725,(0.001,0.5,2)),
 ('F3','cdf',4.69e-20,(1,0,4.69)),('F3','logcdf',4.69e-20,(1,0,4.69)),('F3','logsf',4.69e-20,(1,0,4.69)),
 ('F3','cdf',3e-60,(5,0,3)),('F3','cdf',1e-5,(1,0,4.69)),
 ('F3','cdf',1.000000000099e-10,(0.5,1e-10,1e-8)),('F3','cdf',0.2500000000015,(2.5,0.25,1.75)),
 ('F3','logsf',0.2500000000015,(2.5,0.25,1.75)),
 ('F4','sf',9.99999e-07,(2,0,1e-6)),('F4','logsf',9.99999e-07,(2,0,1e-6)),('F4','logcdf',9.99999e-07,(2,0,1e-6)),
 ('F4','sf',0.00999999999999,(3,0,0.01)),('F4','logsf',9.9999999999901e-09,(0.5,1e-10,1e-8)),
 ('F4','logcdf',15.685,(2.3,15.3,16)),('F4','sf',1.9999999999985,(0.001,0.5,2)),
 ('F4','logcdf',1.7499999999985,(2.5,0.25,1.75)),
 ('F5','ppf',1e-300,(1,0,4.69)),('F5','ppf',1e-17,(1,0,4.69)),('F5','ppf',1e-300,(5,0,3)),
 ('F5','ppf',1e-10,(1,0,4.69)),('F5','isf',0.99999280314327,(2,0,1e-6)),
 ('F5','ppf',3.4807005884285624e-14,(0.5,1e-10,1e-8)),('F5','isf',0.9999218629262349,(0.5,1e-10,1e-8)),
 ('F6','ppf',0.999999938945977,(0.001,0.5,2)),('F6','isf',0.08,(0.001,0.5,2)),
]

def ref(m,x,s,dps):
    with mp.workdps(dps):
        return F[m](M(x),*[M(v) for v in s])

for fid,m,x,s in rows:
    r1, r2 = ref(m,x,s,100), ref(m,x,s,200)
    with mp.workdps(200):
        agree = abs(r1-r2) <= abs(r2)*mp.mpf(10)**-30
    sp = getattr(stats.truncweibull_min, m)(x,*s)
    r = float(r2)
    if np.isfinite(sp) and r != 0:
        u = abs(sp - r)/abs(r)/2.220446049250313e-16
        ustr = '%.2g' % u
    else:
        ustr = 'exact' if sp == r else 'inf'
    extra = ''
    with mp.workdps(60):
        sh=[M(v) for v in s]
        if m in ('cdf','logcdf') and fid in ('F1','F3'):
            # quad of pdf from a to x
            Q = mp.quad(lambda t: pdf(t,*sh), [sh[1], M(x)])
            v = Q if m=='cdf' else mp.log(Q)
            extra = ' quadrel=%.1e' % float(abs(v-ref(m,x,s,60))/abs(v))
        if m in ('ppf','isf'):
            xr = ref(m,x,s,200)
            with mp.workdps(200):
                qq = cdf(xr,*[M(v) for v in s]) if m=='ppf' else sf(xr,*[M(v) for v in s])
                extra = ' invrel=%.1e' % float(abs(qq-M(x))/M(x))
    print(f"{fid} {m}({x!r}, {', '.join(map(repr,s))}) scipy={sp!r} ref={mp.nstr(r2,17)} ulp={ustr} agree30={agree}{extra}")
