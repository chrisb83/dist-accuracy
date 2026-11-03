import mpmath as mp, warnings
from scipy import stats
warnings.simplefilter("ignore")
def r_logsf_small(x,c,d):  # x tiny: cdf = exp(L), logsf = log1p(-cdf)
    x,c,d=map(mp.mpf,(x,c,d)); L=d*(c*mp.log(x)-mp.log1p(x**c)); return mp.log1p(-mp.exp(L))
def r_logcdf_big(x,c,d):  # x > 1: -d*log1p(exp(-c log x))
    x,c,d=map(mp.mpf,(x,c,d)); return -d*mp.log1p(mp.exp(-c*mp.log(x)))
for f,args,m in [(r_logsf_small,(1e-300,2,0.5),'logsf'),(r_logcdf_big,(1400,100,1e8),'logcdf'),(r_logcdf_big,(1300,100,1e6),'logcdf')]:
    mp.mp.dps=100; a=f(*args); mp.mp.dps=200; b=f(*args)
    s=getattr(stats.burr,m)(*args)
    print(m,args,repr(s),mp.nstr(b,17),'agree',abs(a-b)<=abs(b)*mp.mpf('1e-30'),'ulp %.3g'%float(abs(mp.mpf(s)-b)/abs(b)/mp.mpf(2.220446049250313e-16)))
# quad cross-check of cdf / sf, integrating pdf in t = log x, scaled by integrand at the endpoint
def pdf_t(t,c,d):  # pdf(e^t)*e^t = c d e^{-c t} (1+e^{-c t})^{-d-1}
    u=mp.exp(-c*t); return c*d*u*(1+u)**(-d-1)
def quad_sf(x,c,d):
    x,c,d=map(mp.mpf,(x,c,d)); t0=mp.log(x); s=pdf_t(t0,c,d)
    return s*mp.quad(lambda t: pdf_t(t,c,d)/s,[t0,t0+1,t0+10,t0+100,mp.inf])
def quad_cdf(x,c,d):
    x,c,d=map(mp.mpf,(x,c,d)); t0=mp.log(x); s=pdf_t(t0,c,d)
    return s*mp.quad(lambda t: pdf_t(t,c,d)/s,[-mp.inf,t0-100,t0-10,t0-1,t0])
for m,args in [('sf',(1e8,2,3)),('sf',(10,10.5,4.3)),('sf',(1.5,100,1e6)),('cdf',(0.00053969845016714859,100,0.001)),('cdf',(7,3,1e4)),('cdf',(1e-300,100,0.001))]:
    g = quad_sf if m=='sf' else quad_cdf
    mp.mp.dps=50; v=g(*args)
    print('quad',m,args,mp.nstr(v,17),'scipy',repr(getattr(stats.burr,m)(*args)))
