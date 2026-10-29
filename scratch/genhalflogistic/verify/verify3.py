import mpmath as mp
import sys; sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/genhalflogistic/verify")
from verify import pdf
c = 0.7727472780992932
for d in (100, 200):
    mp.mp.dps = d
    x = mp.mpf(1e-300)
    # quad with rescaled variable t = x*u
    cq = x*mp.quad(lambda u: pdf(x*u, c), [0, 1])
    L = mp.log1p(-mp.mpf(c)*x)/mp.mpf(c)
    cf = -mp.expm1(L)/(1+mp.exp(L))
    q = mp.mpf(1e-300); cc = mp.mpf(c)
    xp = -mp.expm1(cc*(mp.log1p(-q)-mp.log1p(q)))/cc
    Lp = mp.log1p(-cc*xp)/cc
    chk = (-mp.expm1(Lp)/(1+mp.exp(Lp)) - q)/q
    print(d, mp.nstr(cq, 25), mp.nstr(cf, 25), "ppf", mp.nstr(xp, 20), "cdfchk", mp.nstr(chk, 3))
