import warnings, numpy as np
from scipy import stats
pts = {1.0:[1e-280,0.5,1,30,700], 2.0:[1e-150,1,5,50], 10.0:[0.1,10,50],
       1e3:[500,999,2000], 1e6:[995000,1e6,1010000], 1e8:[99954000,99900000,1e8]}
qs=[1e-300,1e-8,0.5,1-1e-9]
bad=0; n=0
for a,xs in pts.items():
    for m in ['pdf','logpdf','cdf','sf','logcdf','logsf']:
        for x in xs:
            u=getattr(stats.erlang,m)(x,a); v=getattr(stats.gamma,m)(x,a); n+=1
            if not (u==v or (np.isnan(u) and np.isnan(v))): bad+=1; print(m,x,a,u,v)
    for m in ['ppf','isf']:
        for q in qs:
            u=getattr(stats.erlang,m)(q,a); v=getattr(stats.gamma,m)(q,a); n+=1
            if not u==v: bad+=1; print(m,q,a,u,v)
print('compared',n,'differ',bad)
# non-integer a
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    print('erlang.cdf(0.5, 0.5)=',stats.erlang.cdf(0.5,0.5),'gamma',stats.gamma.cdf(0.5,0.5))
    print('erlang.ppf(0.5, 1e-3)=',stats.erlang.ppf(0.5,1e-3),'gamma',stats.gamma.ppf(0.5,1e-3))
    print([str(x.message)[:80] for x in w])
print('erlang.pdf(1,0.5) nan?', stats.erlang.pdf(1,0.5))
