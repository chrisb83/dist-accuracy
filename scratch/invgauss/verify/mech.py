import numpy as np, warnings
warnings.simplefilter('ignore')
import scipy.special._ufuncs as scu
from scipy import stats
print('isf boost(1e-17,1e-3)', scu._invgauss_isf(1e-17, 1e-3, 1))
try: print(scu._invgauss_ppf(1.0, 1e-3, 1))
except Exception as e: print('ppf(1.0):', type(e).__name__, e)
for mu in (2e-3, 3e-3):
    print(mu, scu._invgauss_ppf(0.3, mu, 1), scu._invgauss_isf(0.3, mu, 1))
print('ppf 1.21e-26 repeated', [stats.invgauss.ppf(1.21e-26, 0.14546264555347513) for _ in range(2)])
print(stats.invgauss.xtol)
