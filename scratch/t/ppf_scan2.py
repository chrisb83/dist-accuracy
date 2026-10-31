import sys
sys.path.insert(0, 'refs')
import numpy as np, mpmath as mp, scipy.special as sc
import t_ref as r
mp.mp.dps = 50
for df in [0.3, 0.7, 1.2, 1.5, 1.9, 2.2, 2.5, 2.74, 2.9, 3.3, 3.7, 4.5, 6.5, 10.5, 20.5]:
    worst = (0, None)
    for q in np.linspace(0.505, 0.995, 50):
        x = sc.stdtrit(df, q)
        D = mp.mpf(df)
        err = (r.cdf(mp.mpf(x), D) - mp.mpf(q)) / r.pdf(mp.mpf(x), D)
        ulp = float(abs(err / x)) / 2.22e-16
        if ulp > worst[0]:
            worst = (ulp, q)
    print(df, 'max ulp %.0f at q=%.3f' % worst)
