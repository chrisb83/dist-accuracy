import mpmath as mp, sys
sys.path.insert(0, '.')
from verify import pdf
x0 = 14120.906486483083; df = 2e8
for dps in (100, 200):
    mp.mp.dps = dps
    x = mp.mpf(x0)
    pts = [x - d for d in (60, 30, 15, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.03, 0)]
    q, err = mp.quad(lambda t: pdf(t, df), pts, error=True, maxdegree=10)
    print(dps, 'quad', mp.nstr(q, 35), mp.nstr(err, 3))
    # lower series: P(a,y) = y^a e^-y / Gamma(a+1) * sum_k y^k/((a+1)...(a+k))
    a = mp.mpf(df)/2; y = x*x/2
    s = term = mp.mpf(1); k = 0
    while term > s*mp.mpf(10)**(-dps-5):
        k += 1; term *= y/(a+k); s += term
    P = mp.exp(a*mp.log(y) - y - mp.loggamma(a+1))*s
    print(dps, 'series', mp.nstr(P, 35), k)
