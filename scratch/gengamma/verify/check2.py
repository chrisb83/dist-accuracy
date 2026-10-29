import mpmath as mp
from verify import P_series, P_quad, cdf, ppf, isf
a=1/3; c=3
for x in (1e-200, 1e-105, 1.4132001038995107e-108):
    for dps in (100,200):
        mp.mp.dps=dps
        A,C,X=mp.mpf(a),mp.mpf(c),mp.mpf(x)
        print(x,dps,"series",mp.nstr(P_series(A,C*mp.log(X)),25),"quad",mp.nstr(P_quad(A,C*A*mp.log(X)),25),
              "gammainc",mp.nstr(mp.gammainc(A,0,X**C,regularized=True),25))
for dps in (100,200):
    mp.mp.dps=dps
    print(dps, mp.nstr(isf(1e-150,a,-3),35), mp.nstr(ppf(1e-150,a,3),35))
