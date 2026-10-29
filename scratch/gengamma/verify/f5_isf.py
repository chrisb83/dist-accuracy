# isf(0.43, 2, 0.001): closed form Q(2, y) = exp(-y)(1+y) = q, x = y**1000
import mpmath as mp
for dps in (100, 200):
    mp.mp.dps = dps
    q = mp.mpf(0.43)
    y = mp.findroot(lambda y: mp.exp(-y)*(1+y) - q, 1.6)
    x = y**(1/mp.mpf(0.001))
    print(dps, mp.nstr(x, 20))
    for cand in ('5.8620126999931134e+281', '5.862012699993715e+281'):
        X = mp.mpf(cand); Y = X**mp.mpf(0.001)
        print('  sf(%s) - q = %s' % (cand, mp.nstr(mp.exp(-Y)*(1+Y) - q, 5)))
