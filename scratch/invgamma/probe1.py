import numpy as np, mpmath as mp
from scipy import stats, special as sc
mp.mp.dps = 50
def lp(x, a):
    x, a = mp.mpf(x), mp.mpf(a)
    return -(a+1)*mp.log(x) - 1/x - mp.loggamma(a)
for a in [1e3, 1e6, 1e8]:
    for k in [0, 1, 3]:
        x = 1/(a+1 + k*np.sqrt(a))
        v = stats.invgamma.logpdf(x, a); r = lp(x, a)
        print(a, k, v, mp.nstr(r, 17), float(abs((v-r)/r))/2.2e-16)
# logsf / logcdf generic
for x, a in [(1e80, 4.0669), (1e-3, 4.0669), (1e300, 0.1)]:
    with np.errstate(all='ignore'):
        print('logsf', x, a, stats.invgamma.logsf(x, a), 'logcdf', stats.invgamma.logcdf(x, a))
