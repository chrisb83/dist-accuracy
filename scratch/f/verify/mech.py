import numpy as np, scipy.special as sc
print('betaincinv(2.5,5000,1e-160)', sc.betaincinv(2.5,5000,1e-160), ' 1e-170', sc.betaincinv(2.5,5000,1e-170))
print('fdtri(5,1e4,1e-160)', sc.fdtri(5,1e4,1e-160))
print('betaincinv(0.5,5000,1e-153)', sc.betaincinv(0.5,5000,1e-153))
print('betaincc(0.5,0.5,1e-16)', sc.betaincc(0.5,0.5,1e-16), ' 1e-20', sc.betaincc(0.5,0.5,1e-20))
print('betaincc(0.5,1,1e-16)', sc.betaincc(0.5,1,1e-16), 1-np.sqrt(1e-16))
r=(0.01/2)/1e308; yc=r/(1+r)
print('F2 safe: betainc(0.005,1,yc)=', sc.betainc(0.005,1.0,yc))
print('fdtrc(1,1,1e-16)', sc.fdtrc(1,1,1e-16))
