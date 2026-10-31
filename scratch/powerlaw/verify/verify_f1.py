import mpmath as mp, numpy as np, scipy, scipy.stats as st
pts = [(1.659113328990585,0.4638806987506132),(0.001,0.00099310918137498),(0.1,0.07742636826811275),
       (3,0.5773502691896257),(1e-12,9.999999999723714e-13),(1e-12,1.0000009999723714e-12),
       (1e-12,9.999989999723714e-13),(0.001,0.0009931081882657985),(0.1,0.07742629084174447)]
def ref(a,x,dps):
    with mp.workdps(dps):
        a=mp.mpf(a); x=mp.mpf(x)
        # route: (a-1)*log(x/x0), x0 = a**(-1/(a-1)) ; cross-check against log of pdf integral derivative-free form
        x0 = mp.exp(-mp.log(a)/(a-1))
        r1 = (a-1)*mp.log(x/x0)
        r2 = mp.log(a*mp.power(x,a-1))
        assert abs(r1-r2) <= abs(r1)*mp.mpf(10)**(-dps+20)
        return r1
print(scipy.__version__)
for a,x in pts:
    r100=ref(a,x,100); r200=ref(a,x,200)
    assert abs(r100-r200) <= abs(r200)*mp.mpf(10)**-30
    s=st.powerlaw.logpdf(x,a)
    ulp = abs(mp.mpf(s)-r200)/(abs(r200)*mp.mpf(2)**-52)
    print(a, repr(x), mp.nstr(r200,17), repr(s), mp.nstr(ulp,3))
