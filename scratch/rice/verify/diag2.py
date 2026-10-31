import mpmath as mp
def i0s(z):
    t = mp.mpf(1); s = t; k = 0; q = z*z/4
    while True:
        k += 1; t = t*q/(k*k); s += t
        if t < s*mp.eps: return s
for d in (100,200):
    mp.mp.dps=d
    for z in (mp.mpf(3e-85), mp.mpf('1e-30'), mp.mpf(0.37), mp.mpf(5), mp.mpf(400)):
        print(d, mp.nstr(z,5), mp.nstr(mp.besseli(0,z)/i0s(z)-1,5))
