# Recheck cdf(0.0005, 1000) = Q(1000, 2000): (i) 1 - Kummer series P at 300/400 dps,
# (ii) mp.quad of the gamma density with finer subdivision at 100/200 dps.
import mpmath as mp
from verify import P
def Qq(a, y):
    a=mp.mpf(a); y=mp.mpf(y)
    g=lambda t: mp.exp((a-1)*mp.log(t)-t-mp.loggamma(a))
    pts=[y+i for i in (0,0.5,1,2,4,8,16,32,64,128,256,512,1024,2048,4096)]+[mp.inf]
    return mp.quad(g, pts)
r=[]
for d in (300,400):
    mp.mp.dps=d; r.append(1-P(1000,1/mp.mpf(0.0005)))
for d in (100,200):
    mp.mp.dps=d; r.append(Qq(1000,1/mp.mpf(0.0005)))
mp.mp.dps=50
for v in r: print(mp.nstr(v,25))
print('series 300 vs 400:', mp.nstr(abs(r[0]-r[1])/r[1],3), ' quad 100 vs 200:', mp.nstr(abs(r[2]-r[3])/r[3],3))
print('ulp scipy:', float(abs(mp.mpf(6.847349459617758e-136)-r[1])/r[1])/2.220446049250313e-16)
