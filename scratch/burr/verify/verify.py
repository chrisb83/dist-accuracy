# Independent reference for burr (Type III), standard form.
# pdf in the x**c form: c d x^(cd-1) (1+x^c)^(-d-1); cdf = exp(d*(c log x - log1p(x^c)));
# sf = -expm1(log cdf); quantiles via expm1 form, checked by cdf(x_ref) == q.
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")

def logcdf(x,c,d): return d*(c*mp.log(x) - mp.log1p(x**c))
def ref(m, a, c, d):
    a, c, d = mp.mpf(a), mp.mpf(c), mp.mpf(d)
    if m == 'pdf': return c*d*x_pow(a, c*d-1) / (1+a**c)**(d+1)
    if m == 'logpdf': return mp.log(c)+mp.log(d)+(c*d-1)*mp.log(a)-(d+1)*mp.log1p(a**c)
    if m == 'cdf': return mp.exp(logcdf(a,c,d))
    if m == 'logcdf': return logcdf(a,c,d)
    if m == 'sf': return -mp.expm1(logcdf(a,c,d))
    if m == 'logsf': return mp.log(-mp.expm1(logcdf(a,c,d)))
    if m == 'ppf':
        x = mp.expm1(-mp.log(a)/d)**(-1/c); assert abs(mp.exp(logcdf(x,c,d))/a-1) < mp.mpf(10)**(-mp.mp.dps+20); return x
    if m == 'isf':
        x = mp.expm1(-mp.log1p(-a)/d)**(-1/c); s = -mp.expm1(logcdf(x,c,d)); assert abs(s/a-1) < mp.mpf(10)**(-mp.mp.dps+20); return x
def x_pow(a,e): return a**e

calls = [
 ('F1','sf',1e8,2,3),('F1','logsf',1e8,2,3),('F1','sf',10,10.5,4.3),('F1','logsf',10,10.5,4.3),('F1','sf',1e10,1,1),('F1','sf',1.5,100,1e6),
 ('F2','logpdf',1e-200,2,3),('F2','pdf',1e-160,2,0.6),('F2','pdf',1e-60,2,3),('F2','pdf',1e-300,0.5,0.001),('F2','logcdf',1e-200,2,3),
 ('F2','cdf',1e-300,100,0.001),('F2','cdf',0.00053969845016714859,100,0.001),('F2','sf',1e-70,100,0.001),('F2','sf',1e-20,100,0.001),('F2','logsf',1e-300,2,0.5),
 ('F3','logpdf',1e-10,2,0.5),('F3','logpdf',1e-100,2,0.5),('F3','logpdf',1e-10,1,1),('F3','logpdf',1e-10,2,0.51),
 ('F4','ppf',0.9999999999999997,2,3),('F4','ppf',0.99999999,3,1e6),('F4','ppf',0.99999999948205254,3,1e4),('F4','ppf',0.5,3,1e6),('F4','ppf',0.99,0.01,10),
 ('F5','ppf',1e-300,2,0.5),('F5','ppf',0.42999999999999994,100,0.001),('F5','ppf',0.01,100,0.001),
 ('F6','isf',0.57,100,0.001),('F6','isf',0.99,100,0.001),('F6','isf',0.99999999999999967,100,0.001),
 ('F7','pdf',1300,100,1e6),('F7','pdf',1275.1256477602437,100,1e6),('F7','logcdf',1400,100,1e8),('F7','logcdf',1300,100,1e6),
 ('F8','pdf',100,3,1e6),('F8','cdf',1e3,3,1e4),('F8','cdf',1.5,100,1e6),('F8','cdf',7,3,1e4),('F8','logsf',8.5,3,1e4),
]
for fid,m,a,c,d in calls:
    mp.mp.dps=100; r1=ref(m,a,c,d)
    mp.mp.dps=200; r2=ref(m,a,c,d)
    agree = r1==r2==0 or abs(r1-r2) <= abs(r2)*mp.mpf('1e-30')
    s = getattr(stats.burr, m)(a, c, d)
    rf = float(r2)
    if rf == 0: ulp = 0.0 if s == 0 else float('inf')
    elif not np.isfinite(s): ulp = float('inf')
    else: ulp = float(abs(mp.mpf(s)-r2)/abs(r2)/mp.mpf(2.220446049250313e-16))
    print(f"{fid}|{m}({a!r}, {c}, {d})|{s!r}|{mp.nstr(r2,17)}|{ulp:.3g}|agree={agree}")
