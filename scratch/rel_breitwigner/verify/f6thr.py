import numpy as np, warnings
warnings.simplefilter('ignore')
from scipy import stats
M = np.finfo(float).max
for rho in [1.4e154, 1.6e154, 2e154, 3e154, 3.3e154, 3.4e154, 4e154]:
    d = 1.0001*(M/2)/rho/2      # (x-rho)(x+rho) = d*(2 rho + d) ~ 2 rho d just past DBL_MAX
    d = 1.0001*(M/2)/rho
    x = rho + d
    u = d*((2*rho + d)/rho)
    true = (2/np.pi)/u/u
    print(rho, "x-rho=", d, "scipy", stats.rel_breitwigner.pdf(x, rho), "approx true", true)
