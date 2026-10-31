# Bulk scan of stdtrit: error in x estimated as (F(x_scipy) - q)/f(x) at 60 digits.
import sys
sys.path.insert(0, 'refs')
import numpy as np, mpmath as mp, scipy.special as sc
import t_ref as r
mp.mp.dps = 60
for df in [2.7433514990818093, 3.0, 5.5, 30.0]:
    worst = (0, None)
    for q in np.linspace(0.51, 0.99, 49):
        x = sc.stdtrit(df, q)
        D = mp.mpf(df)
        err = (r.cdf(mp.mpf(x), D) - mp.mpf(q)) / r.pdf(mp.mpf(x), D)
        ulp = float(abs(err / x)) / 2.22e-16
        if ulp > worst[0]:
            worst = (ulp, q)
    print(df, 'max ulp %.0f at q=%.2f' % worst)
