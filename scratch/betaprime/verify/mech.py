import mpmath as mp, scipy.special as sc, warnings
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps=60
for a,b in [(1e5,3e5),(1000,0.5)]:
    e=sc.betaln(a,b)-mp.log(mp.beta(a,b)); print("betaln err",a,b,mp.nstr(e,3))
print("beta._ppf(1e-120,2,10)",stats.beta._ppf(1e-120,2,10), "beta._ppf(1e-200,5,2)",stats.beta._ppf(1e-200,5,2))
# F4: r from scipy vs exact r = x/(1+x)
for q,a,b,xref in [(0.6546077528522537,1000,0.5,mp.mpf('9989.999999999996')),(0.008176429292048043,2,0.001,mp.mpf('9989.999999999995'))]:
    r=stats.beta._ppf(q,a,b); rex=xref/(1+xref)
    print("r relerr ulp",mp.nstr(abs(mp.mpf(r)-rex)/rex/2**-52,3),"r/(1-r)",r/(1-r))
