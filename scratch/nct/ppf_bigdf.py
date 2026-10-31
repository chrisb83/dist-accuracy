import sys, mpmath as mp, warnings
sys.path.insert(0, "refs"); import nct_ref as R
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 50
for k in [1e4, 1e6]:
    for q in [1e-300, 1e-100, 1e-30, 1e-20, 1e-10]:
        r = R.ppf(q, k, 2.0); s = stats.nct.ppf(q, k, 2.0)
        print(k, q, s, mp.nstr(r, 15), "cdf(scipy ppf) =", stats.nct.cdf(s, k, 2.0), mp.nstr(R.cdf(s,k,2.0),6))
