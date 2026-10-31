import mpmath as mp
mp.mp.dps=100
b=mp.mpf(1e-5); x=mp.mpf(3e-80)
bad=[]
def f(u):
    t=u*x
    r = mp.besseli(0,t*b)
    if r != 1: bad.append((u, r-1))
    return r
mp.quad(f,[0,1])
print(len(bad), bad[:3])
