import mpmath as mp
from scipy import stats
import sys; sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/genhalflogistic/verify")
from verify import pdf, cdf_quad
for d in (100, 200):
    mp.mp.dps = d
    print(mp.nstr(mp.mpf(1e-300), 25), mp.nstr(cdf_quad(1e-300, 0.7727472780992932), 25))
    q = mp.mpf(1e-300); c = mp.mpf(0.7727472780992932)
    # ppf via series-safe log: log t = log1p(-q) - log1p(q)
    x = -mp.expm1(c*(mp.log1p(-q)-mp.log1p(q)))/c
    t = mp.exp(mp.log1p(-c*x)/c)
    print("ppf", mp.nstr(x, 20), "cdfchk", mp.nstr(((1-t)/(1+t)-q)/q, 3), "scipy", stats.genhalflogistic.ppf(1e-300, 0.7727472780992932))
