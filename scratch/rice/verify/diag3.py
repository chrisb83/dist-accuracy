import mpmath as mp
for d in (100,200):
    mp.mp.dps=d
    b=mp.mpf(1e-5); x=mp.mpf(3e-80)
    print(d, mp.nstr(mp.quad(lambda u: u,[0,1])-mp.mpf(1)/2,5), mp.nstr(mp.exp(-b*b/2),25), mp.nstr(x*x/2*mp.exp(-b*b/2),30))
    print(d, mp.nstr(mp.quad(lambda u: u*mp.exp(-b*b/2),[0,1])-mp.exp(-b*b/2)/2,5), mp.nstr(mp.quad(lambda u: u*mp.exp(-(u*u+b*b)/2),[0,1]),30))
