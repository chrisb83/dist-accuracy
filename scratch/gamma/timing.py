import time, mpmath as mp
mp.mp.dps = 400
def t(f, *args):
    t0=time.time(); v=f(*args); return time.time()-t0, mp.nstr(v,12)
cases = [(0.001,1e-300),(0.001,1.0),(1.9932,1.0),(1.9932,700.0),(1e6,1e6),(1e6,1.1e6),(1e6,1e3),(1e6,1e300),(1e-3,1e300),(2.0,1e300),(1e300,1e300)]
for a,x in cases:
    a=mp.mpf(a); x=mp.mpf(x)
    try:
        d1,v1=t(lambda: mp.gammainc(a,0,x,regularized=True))
    except Exception as e:
        d1,v1=-1,repr(e)[:40]
    try:
        d2,v2=t(lambda: mp.gammainc(a,x,mp.inf,regularized=True))
    except Exception as e:
        d2,v2=-1,repr(e)[:40]
    print(f"a={float(a):.3g} x={float(x):.3g} lower {d1:7.3f}s {v1}   upper {d2:7.3f}s {v2}")
