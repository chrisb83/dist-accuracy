# Mechanism checks: F1 (2000-term partial Kummer series), F3 (term rounding)
import mpmath as mp, numpy as np, scipy.special as sc
mp.mp.dps = 50
for df, x in [(2e8, 199908000.), (2e6, 1990800.)]:
    a = mp.mpf(df)/2; y = mp.mpf(x)/2
    pref = mp.exp(a*mp.log(y) - y - mp.loggamma(a+1))
    s = mp.mpf(1); t = mp.mpf(1)
    for n in range(1, 2000):
        t *= y/(a+n); s += t
    print(df, 'partial2000=', mp.nstr(pref*s, 17), 'chdtr=', repr(sc.chdtr(df, x)), 'gammainc=', repr(sc.gammainc(df/2, x/2)))
# F3: logpdf with exact terms but each rounded to double
df, x = 2e8, 2e8
terms = [sc.xlogy(df/2-1, x), x/2, sc.gammaln(df/2), np.log(2)*df/2]
print('terms', terms, 'ulp(max term)=', np.spacing(max(terms)))
