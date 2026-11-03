import mpmath as mp
# cdf(-1,1000) via Watson's lemma: F = e^{k-2k u0}/(2 pi I0(k)) * sum_n g_n n!/(2k)^{n+1},
# g(v) = ((u0+v)(1-u0-v))^{-1/2}; plus quad on v in [0, 1-u0] with nodes at multiples of 1/k
def both(x,k,N):
    x=mp.mpf(x);k=mp.mpf(k);u0=mp.sin(x/2)**2
    g=lambda v: 1/mp.sqrt((u0+v)*(1-u0-v))
    c=mp.taylor(g,0,N)
    S=mp.fsum(c[n]*mp.factorial(n)/(2*k)**(n+1) for n in range(N+1))
    pref=mp.exp(k-2*k*u0)/(2*mp.pi*mp.besseli(0,k))
    nodes=[mp.mpf(0)]+[mp.mpf(j)/k for j in (1,2,4,8,16,32,64,128)]+[1-u0]
    Q,err=mp.quad(lambda v: mp.exp(-2*k*v)*g(v),nodes,error=True,maxdegree=10)
    return pref*S, pref*Q, err/Q, abs(c[N]*mp.factorial(N)/(2*k)**(N+1)/S)
for dps,N in ((100,120),(200,200)):
    mp.mp.dps=dps
    w,q,e,last=both(-1.0,1000,N)
    print(dps,'watson',mp.nstr(w,25),'quad',mp.nstr(q,25),'quad_err',mp.nstr(e,3),'last_term',mp.nstr(last,3),flush=True)
