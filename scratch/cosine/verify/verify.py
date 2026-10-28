import mpmath as mp, numpy as np, scipy, scipy.stats as st
print(scipy.__version__)
xs_pdf=[3.141592653589793,3.141592643589793,3.141592609180872,3.141582653589793,3.141492653589793,3.1405926535897932,3.1315926535897933]
def pdf(x): return mp.cos(x/2)**2/mp.pi          # half-angle route
def cdf(x):  # quad of pdf from -pi
    return mp.quad(lambda t: (1+mp.cos(t))/(2*mp.pi), [-mp.pi, x])
def ref(f,x):
    out=[]
    for d in (100,200):
        mp.mp.dps=d; out.append(f(mp.mpf(x)))
    mp.mp.dps=100
    assert abs(out[0]-out[1])<=abs(out[1])*mp.mpf(10)**-30, (x,out)
    return out[1]
def ulp(s,r):
    r=float(r) if abs(r)>1e-300 else r
    return abs(mp.mpf(s)-r)/abs(r)/2.220446049250313e-16
for x in xs_pdf:
    for xx in (x,-x):
        r=ref(pdf,xx); s=st.cosine.pdf(xx); print('pdf',xx,s,mp.nstr(r,17),mp.nstr(ulp(s,r),3))
        rl=ref(lambda t: mp.log(pdf(t)),xx); s=st.cosine.logpdf(xx); print('logpdf',xx,s,mp.nstr(rl,17),mp.nstr(ulp(s,rl),3))
r=ref(cdf,-np.pi); print('cdf',st.cosine.cdf(-np.pi),mp.nstr(r,17))
r=ref(lambda t: 1-cdf(t),np.pi); print('sf',st.cosine.sf(np.pi),mp.nstr(r,17))
# closed-form cross check
mp.mp.dps=200; x=mp.mpf(-np.pi); print('closed',mp.nstr((x+mp.pi+mp.sin(x))/(2*mp.pi),17))
nx=np.nextafter(-np.pi,0); mp.mp.dps=100; print('cdf next',st.cosine.cdf(nx),mp.nstr(ref(cdf,nx),17))
from scipy.special import _ufuncs as scu
print('scu', scu._cosine_cdf(-np.pi), st.cosine._cdf(np.array(-np.pi)), st.cosine._sf(np.array(np.pi)))
