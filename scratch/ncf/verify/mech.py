import warnings; warnings.simplefilter('ignore')
from scipy import stats
import numpy as np
print('F1', stats.ncf.sf(10,2,3,0), -stats.ncf.cdf(10,2,3,0))
for s in [(1,1,1),(0.5,1.5,1),(1,1.0000001,1),(2,2,1)]: print('F2', s, stats.ncf.cdf(1e-210,*s))
print('F4', stats.ncf.pdf(8e15,27,27,0.41578441799226107), stats.ncf.pdf(1e16,27,27,0.41578441799226107))
print('F5 clamp', stats.ncf.isf(1e-10,1,1,1) == 2.0**53-1)
print('F8', stats.ncf.sf(1.7e306,1,0.01,1), stats.ncf.sf(1.9e306,1,0.01,1))
