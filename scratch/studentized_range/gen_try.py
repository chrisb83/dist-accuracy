import mpmath as mp, time, sys
def D(z, w):
    r2 = mp.sqrt(2)
    if z + w/2 > 0:
        return (mp.erfc(z/r2) - mp.erfc((z+w)/r2))/2
    return (mp.erfc(-(z+w)/r2) - mp.erfc(-z/r2))/2
def phi(z): return mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
def Rcdf(w, k):
    return k*mp.quad(lambda z: phi(z)*D(z, w)**(k-1), [-mp.inf, -w/2-8, -w/2, -w/2+8, mp.inf])
def Rpdf(w, k):
    return k*(k-1)*mp.quad(lambda z: phi(z)*phi(z+w)*D(z, w)**(k-2), [-mp.inf, -w/2-8, -w/2, -w/2+8, mp.inf])
def logchi(s, df):
    return (df/2)*mp.log(df) - mp.loggamma(df/2) - (df/2-1)*mp.log(2) + (df-1)*mp.log(s) - df*s*s/2
def outer(g, df):
    a = min(mp.mpf('0.9'), 12/mp.sqrt(2*df))
    pts = [0, 1-a, 1, 1+a, mp.inf]
    return mp.quad(lambda s: mp.exp(logchi(s, df))*g(s), pts)
def cdf(q,k,df): return outer(lambda s: Rcdf(q*s,k), df)
def pdf(q,k,df): return outer(lambda s: s*Rpdf(q*s,k), df)
mp.mp.dps = int(sys.argv[1])
for f,args in [(cdf,(1,2,10)),(pdf,(1,2,10)),(cdf,(1,3,10)),(pdf,(1,3,10)),(pdf,(1,1.5,10))]:
    t=time.time(); v=f(*map(mp.mpf,args)); print(f.__name__,args,mp.nstr(v,20),round(time.time()-t,1))
