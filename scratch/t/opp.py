import sys, warnings
sys.path.insert(0, 'refs')
import mpmath as mp, numpy as np
import t_ref as r
from scipy.stats import t
mp.mp.dps = 100
warnings.simplefilter('ignore')
for x, df in [(-1e120, 2.7433514990818093), (-1e10, 30.0), (40.0, 30.0)]:
    ref_c = mp.log(r.cdf(mp.mpf(x), mp.mpf(df)))
    ref_s = mp.log(r.sf(mp.mpf(x), mp.mpf(df)))
    print(x, df, 'logcdf', t.logcdf(x, df), mp.nstr(ref_c, 17), 'logsf', t.logsf(x, df), mp.nstr(ref_s, 17))
