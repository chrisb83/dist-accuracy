# Independent check of refs/f_ref.py where the harness integral self-check FAILs
# for (0.01, 3): cdf vs mp.betainc, and pdf integral in log-x substitution.
import sys; sys.path.insert(0, "refs")
import mpmath as mp, f_ref as r
mp.mp.dps = 60
n, m = mp.mpf('0.01'), mp.mpf(3)
for x in ['1.6269e-198', '1.01242e-58', '1.14779e-7']:
    x = mp.mpf(x); t = n*x/m; y = t/(1+t)
    print(mp.nstr(r.cdf(x, n, m), 30), mp.nstr(mp.betainc(n/2, m/2, 0, y, regularized=True), 30))
a, b = mp.mpf('1.6269e-198'), mp.mpf('1.01242e-58')
I = mp.quad(lambda u: r.pdf(mp.exp(u), n, m)*mp.exp(u), [mp.log(a), mp.log(b)])
print('quad logx', mp.nstr(I, 25), 'cdf diff', mp.nstr(r.cdf(b, n, m) - r.cdf(a, n, m), 25))
