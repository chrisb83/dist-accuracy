import mpmath as mp
for dps in (100,200):
    mp.mp.dps=dps
    x=mp.mpf(1e-312); b=mp.mpf(1e-12)
    num = x*mp.quad(lambda u: mp.exp(-x*u), [0,1])
    den = b*mp.quad(lambda u: mp.exp(-b*u), [0,1])
    naive = mp.quad(lambda t: mp.exp(-t), [0,x])/mp.quad(lambda t: mp.exp(-t), [0,b])
    print(dps, mp.nstr(num/den,20), mp.nstr(naive,20), mp.nstr(mp.quad(lambda t: mp.exp(-t), [0,x])/x,25))
