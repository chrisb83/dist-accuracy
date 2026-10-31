import numpy as np, warnings
warnings.simplefilter('ignore')
from scipy import stats
d = np.float64
# F1: -rho*(rho+1j) at 1.34e154 / 1.35e154 / 1e160
for rho in [1.34e154, 1.35e154, 1e160]:
    w = -d(rho)*(d(rho)+1j); b = np.sqrt(w)
    print("F1", rho, w, b, d(1e20)/b, stats.rel_breitwigner.cdf(1e20, rho))
# F2 boundaries
for rho in [8.62e153, 8.63e153, 8.6e153, 8.65e153]:
    print("F2", rho, stats.rel_breitwigner.cdf(3.5, rho), "approx true", 2*3.5/(np.pi*rho**2))
# F3: C in _cdf
for rho in [7.45e-155, 7.47e-155, 1e-160]:
    rho = d(rho); C = np.sqrt(2/(1 + np.sqrt(1 + 1/rho**2)))/np.pi
    print("F3", rho, 1/rho**2, C, stats.rel_breitwigner.cdf(1, rho))
# F4: C in _pdf
for rho in [1.06e-154, 1.05e-154, 1e-154, 7e-155, 1e-160]:
    rho = d(rho)
    num = 2*(1 + 1/rho**2); den = 1 + np.sqrt(1 + 1/rho**2)
    print("F4", rho, num, den, np.sqrt(num/den)*2/np.pi, stats.rel_breitwigner.pdf(rho, rho))
# F5: u and u**2
for x, rho in [(127.6, 1e-150), (1e20, 1e-150), (115., 1e-150)]:
    u = (d(x)-rho)*(d(x)+rho)/rho
    print("F5", x, rho, u, u**2, stats.rel_breitwigner.pdf(x, rho))
# F6
rho = d(1e160); x = d(9.9999999e159)
print("F6", (x-rho)*(x+rho), (x-rho)*((x+rho)/rho))
for rho in [1.5e154, 1.6e154, 1.7e154, 1e155]:
    rho = d(rho); x = rho*(1-1e-8)
    print("F6 thr", rho, stats.rel_breitwigner.pdf(x, rho), (x-rho)*(x+rho))
