import numpy as np, scipy.special as sc, mpmath as mp, warnings
warnings.simplefilter('ignore')
print('2**-47', 2.0**-47, sc.tklmbda(-1e300, -1.0))
print('logistic', 1/(1+np.exp(10.0)), sc.tklmbda(-10.0, 1e-5))
for lam in (9.99e-5, 1e-4, 1.0001e-4):
    print('thr', lam, sc.tklmbda(-3.0, lam), 1/(1+np.exp(3.0)))
print('F4 tklmbda(100,0)=', sc.tklmbda(100.0, 0.0))
print('F5 terms', sc.boxcox(0.49, 50.0), sc.boxcox1p(-0.49, 50.0))
lam = 3.1321477856738267; x = 0.3192697370711285
print('F6', 1/lam == x, mp.mpf(x) < 1/mp.mpf(lam), sc.tklmbda(-x, lam))
