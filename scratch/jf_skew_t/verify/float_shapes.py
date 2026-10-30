import warnings; warnings.simplefilter('ignore')
import mpmath as mp, numpy as np
from scipy import stats
import scipy.special as sc
from verify import pdf
for x,a,b in [(3.0,600.0,600.0),(-1e-10,600.0,600.0),(1e-10,1000.0,5.0),(0.5,32.0,32.0),(0.5,40.0,40.0)]:
    r = {}
    for d in (100,200):
        mp.mp.dps=d; r[d]=pdf(mp.mpf(x),mp.mpf(a),mp.mpf(b))
    s = stats.jf_skew_t.pdf(x,a,b)
    ulp = float(abs(mp.mpf(float(s))-r[200])/r[200])/2.22e-16 if np.isfinite(s) else float('inf')
    print(x,a,b,repr(s),mp.nstr(r[200],17),f"{ulp:.3g}", abs(r[100]-r[200])<=r[200]*mp.mpf(10)**-30)
mp.mp.dps=50
print('sc.beta(1000,5) ulp', float(abs(mp.mpf(sc.beta(1000.,5.))-mp.beta(1000,5))/mp.beta(1000,5))/2.22e-16)
print('2**(1199.)', 2.0**1199., 'sc.beta(600,600)', sc.beta(600.,600.))
