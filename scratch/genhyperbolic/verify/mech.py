import warnings; warnings.simplefilter('ignore')
import numpy as np
from scipy import special as sc, integrate
from scipy.stats import _stats
def naive(x,p,a,b):
    g=np.sqrt((a+b)*(a-b)); s=np.sqrt(1+x*x)
    lc = 0.5*p*np.log(g*g) - (np.log(sc.kve(p,g)) - g) - 0.5*np.log(2*np.pi) - (p-0.5)*np.log(a)
    lx = (np.log(sc.kve(p-0.5,a*s)) - a*s) + (p-0.5)*np.log(s) + b*x
    return np.exp(lc+lx)
for args in [(0,1,1e9,0),(1e9,-1.5,1,0.9999999999),(0.002,1,1e6,0)]:
    print(args, repr(naive(*args)), repr(_stats.genhyperbolic_pdf(*args)))
print('exact-combination pdf(0,1,1e9,0):', repr(np.sqrt(1e9/(2*np.pi))*sc.kve(0.5,1e9)/sc.kve(1,1e9)))
f=lambda x: _stats.genhyperbolic_pdf(x,1,1e6,0)
print('quad(-inf,-0.008) a=1e6:', integrate.quad(f,-np.inf,-0.008,epsrel=1e-10,epsabs=0,full_output=1)[:2])
print('pdf(-1100,1,1e6,0)=', _stats.genhyperbolic_pdf(-1100.,1,1e6,0), 'pdf(-1000)=',_stats.genhyperbolic_pdf(-1000.,1,1e6,0))
h=lambda x: _stats.genhyperbolic_pdf(x,-1.5,1,0.9999999999)
print('quad(1e6,inf) heavy:', integrate.quad(h,1e6,np.inf,epsrel=1e-10,epsabs=0)[:2])
print('quad(1e4,inf) heavy:', integrate.quad(h,1e4,np.inf,epsrel=1e-10,epsabs=0)[:2])
print('min(1.0,nan)=', min(1.0,float('nan')), 'g=0 pieces:', sc.kve(-1,0.0), np.log(0.0))
