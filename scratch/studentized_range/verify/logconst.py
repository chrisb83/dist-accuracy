import mpmath as mp
from scipy.stats import _stats
mp.mp.dps=50
k,df=3,99999
ex = mp.log(k) + (mp.mpf(df)/2)*mp.log(df) - mp.loggamma(mp.mpf(df)/2) - (mp.mpf(df)/2-1)*mp.log(2)
v=_stats._studentized_range_cdf_logconst(k,df)
print(repr(v), mp.nstr(ex,20), mp.nstr(mp.mpf(v)-ex,3))
