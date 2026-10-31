import mpmath as mp
def pdf(x, b):
    return x*mp.exp(-(x*x+b*b)/2)*mp.besseli(0, x*b)
for d in (100,200):
    mp.mp.dps=d
    x=mp.mpf(3e-80); b=mp.mpf(1e-5)
    print(d, mp.nstr(mp.besseli(0,x*b)-1,20), mp.nstr(mp.quad(lambda t: pdf(t,b),[0,x]),30), mp.nstr(mp.quad(lambda u: pdf(u*x,b)*x,[0,1]),30))
    x=mp.mpf(10); b=mp.mpf(40)
    print(d, mp.nstr(mp.quad(lambda t: pdf(t,b),mp.linspace(0,x,9)),30), mp.nstr(mp.quad(lambda t: pdf(t,b),mp.linspace(0,x,41)),30))
