# arbitration: closed form Phi(x) - exp(1/(2K^2) - x/K) Phi(x - 1/K) at 200 digits,
# plus layer integral with mp.quad error estimate and a denser split
import mpmath as mp
mp.mp.dps = 200
Phi = lambda z: mp.erfc(-z/mp.sqrt(2))/2
def closed(x, K):
    x = mp.mpf(x); K = mp.mpf(K)
    return Phi(x) - mp.exp(1/(2*K**2) - x/K)*Phi(x - 1/K)
def layer(x, K, n):
    x = mp.mpf(x); K = mp.mpf(K)
    f = lambda s: mp.npdf(s) * (-mp.expm1(-(x - s) / K))
    pts = [x - mp.mpf(12)*i/n for i in range(n, -1, -1)]
    return mp.quad(f, pts, error=True)
for x, K in [(-37.0, 1.0), (-37.0, 1.5), (-37.0, 0.1), (-30.0, 10), (-37.0, 1e4), (-20.0, 1e8)]:
    c = closed(x, K)
    l, e = layer(x, K, 400)
    print(x, K, mp.nstr(c, 20), mp.nstr(l, 20), "quad err", mp.nstr(e, 3))
