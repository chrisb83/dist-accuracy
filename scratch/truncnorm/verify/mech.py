import numpy as np, scipy.special as sc
from scipy.stats._continuous_distns import _log_diff, _log_gauss_mass
from scipy import stats
print('exp(1j*pi)', np.exp(1j*np.pi))
print('_log_diff(-0.5,-0.5)', _log_diff(-0.5, -0.5))
p = sc.log_ndtr(-30.0)
print('mass(30,inf)==log_ndtr(-30):', _log_gauss_mass(np.array(30.), np.array(np.inf)) == p,
      ' log1p(-1e-20)+p==p:', np.log1p(-1e-20) + p == p)
print('-ndtri_exp(p-36.6387...)', -sc.ndtri_exp(p + np.log(1.2246467991473532e-16)))
print('isf(1e-20,30,inf), isf(1e-25,30,inf):', stats.truncnorm.isf([1e-20, 1e-25], 30, np.inf))
print('central-case for a=0: log1p(-ndtr(0)-ndtr(-1e-20))', sc.log1p(-sc.ndtr(0.) - sc.ndtr(-1e-20)))
