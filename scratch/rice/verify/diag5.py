import mpmath as mp
mp.mp.dps=100
b=mp.mpf(1e-5); x=mp.mpf(3e-80)
g1=lambda u: u*mp.exp(-(u*u*x*x+b*b)/2)
g2=lambda u: u*mp.exp(-(u*u*x*x+b*b)/2)*mp.besseli(0,u*x*b)
g3=lambda u: u*mp.exp(-b*b/2)
for g in (g1,g2,g3):
    print(mp.nstr(mp.quad(g,[0,1])*x*x,30), mp.nstr(mp.quad(g,[0,1],error=True)[1],5))
print(mp.nstr(mp.exp(-(x*x+b*b)/2),30), mp.nstr(mp.exp(-b*b/2),30))
