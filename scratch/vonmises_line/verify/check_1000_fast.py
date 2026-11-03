import mpmath as mp
# substitution u = sin^2(t/2): 1-cos t = 2u; integrand exp(-2k u) / sqrt(u(1-u)) du over u in [sin^2(x/2),1] (for x<0, t from -pi to x)
def cdf_u(x,k):
    x=mp.mpf(x);k=mp.mpf(k); u0=mp.sin(x/2)**2
    f=lambda u: mp.exp(-2*k*(u-u0))/mp.sqrt(u*(1-u))
    nodes=[u0+(1-u0)*mp.mpf(2)**(-i) for i in range(60,0,-1)]
    I=mp.quad(f,[u0]+nodes+[1])
    return I*mp.exp(k-2*k*u0)/(2*mp.pi*mp.besseli(0,k))
for x,k in [(-1.0,1000),(-3.0,200)]:
    for dps in (100,200):
        mp.mp.dps=dps; print(x,k,dps,mp.nstr(cdf_u(x,k),20),flush=True)
