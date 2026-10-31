# Independent pdf/cdf consistency check for df = 0.001 on [-2.11996e152, 0],
# integrating in u = log|x| (the harness quad in x cannot resolve this range).
import sys
sys.path.insert(0, 'refs')
import mpmath as mp
import t_ref as r
mp.mp.dps = 30
df = mp.mpf(0.001)
X = mp.mpf(-2.1199605744342963e152)
U = mp.log(-X)
I = mp.quad(lambda u: r.pdf(-mp.exp(u), df) * mp.exp(u),
            [-200, -50, -10, 0, 10, 50, 100, 200, U])
I += mp.quad(lambda x: r.pdf(x, df), [-mp.exp(-200), 0])
D = r.cdf(mp.mpf(0), df) - r.cdf(X, df)
print(mp.nstr(I, 20), mp.nstr(D, 20), mp.nstr(I / D - 1, 3))
