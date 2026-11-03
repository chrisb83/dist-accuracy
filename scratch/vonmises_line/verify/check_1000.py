import mpmath as mp
# resolve cdf(-1,1000) and cdf(-3,200): quad with nodes geometrically clustered at the upper endpoint x
def cdf_quad_geo(x,k):
    x=mp.mpf(x);k=mp.mpf(k)
    f=lambda t: mp.exp(k*(mp.cos(t)-1))
    a=-mp.pi
    d=[mp.mpf(2)**(-i) for i in range(0,40)]
    nodes=sorted(set([a]+[x-(x-a)*di for di in d[1:]]+[x]))
    return mp.quad(f,nodes)/(2*mp.pi*mp.besseli(0,k)*mp.exp(-k))
def cdf_series(x,k):
    x=mp.mpf(x);k=mp.mpf(k);i0=mp.besseli(0,k)
    s=(x+mp.pi)/(2*mp.pi)
    for j in range(1,4000):
        s+=mp.besseli(j,k)/i0*mp.sin(j*x)/(j*mp.pi)
    return s
for x,k in [(-1.0,1000),(-3.0,200)]:
    for dps in (100,200):
        mp.mp.dps=dps; print(x,k,dps,'geo',mp.nstr(cdf_quad_geo(x,k),20))
    mp.mp.dps=600; print(x,k,600,'series',mp.nstr(cdf_series(x,k),20))
