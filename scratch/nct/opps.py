import sys, mpmath as mp, warnings
sys.path.insert(0, "refs"); import nct_ref as R
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 60
k, c = 14.0, 0.24045031331198066
print("logpdf(1e30)", stats.nct.logpdf(1e30, k, c), mp.nstr(R.REFS["logpdf"](1e30, k, c), 17))
print("logsf(1e30)", stats.nct.logsf(1e30, k, c), mp.nstr(R.REFS["logsf"](1e30, k, c), 17))
print("logcdf(-1e30)", stats.nct.logcdf(-1e30, k, c), mp.nstr(R.REFS["logcdf"](-1e30, k, c), 17))
