import warnings, numpy as np
warnings.simplefilter('ignore')
from scipy.stats import genhyperbolic as G
import scipy; print(scipy.__version__)
B=0.9999999999
calls=[('pdf',0,-1,1,1),('pdf',-200,-1,1,1),('pdf',500,-1,1,1),('logpdf',0,-1,1,1),('cdf',0,-1,1,1),('cdf',-2,-1,1,1),('cdf',-200,-1,1,1),('sf',10,-1,1,1),
('logpdf',1e9,0.5,1.5,-0.5),('logpdf',1.01*2**30/2,1,2,0),('logpdf',1e300,1,2,0),('logpdf',1e20,0.5,0.001,0),('logpdf',-1e300,-2,3,-2),('pdf',0,1,1.1e9,0),('logpdf',0,1,1.1e9,0),('cdf',0,1,1.1e9,0),
('pdf',0,100,0.01,0),('pdf',0,81,0.01,0),('pdf',0,-100,0.01,0),('logpdf',0,100,0.01,0),('cdf',3,100,0.01,0),('sf',0,100,0.01,0),
('sf',1e4,-1.5,1,B),('cdf',-0.008,1,1e6,0),('sf',1e6,-1.5,1,B),('sf',1e12,-1.5,1,B),('cdf',1e9,-1.5,1,B),('sf',0.008,1,1e6,0),
('pdf',0,1,1e9,0),('pdf',1e9,-1.5,1,B),('pdf',1e6,-1.5,1,B),('pdf',0.002,1,1e6,0),('pdf',-100,-2,3,-2),
('cdf',-0.002,1,1e6,0),('cdf',0.0005,1,1e6,0),('sf',1000,-1.5,1,B)]
for m,*a in calls:
    print(m, a, repr(float(getattr(G,m)(*a))), flush=True)
from scipy import special as sc
print('kve', [sc.kve(0.5, 2.0**30-0.5), sc.kve(0.5, 2.0**30)])
print('kve(81,.01)', sc.kve(81,0.01), 'kv(1,1e6)', sc.kv(1,1e6))
