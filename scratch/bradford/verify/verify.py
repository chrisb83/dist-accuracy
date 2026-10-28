import mpmath as mp, numpy as np, scipy, scipy.stats as st
def pdf(x,c): return c/(mp.log1p(c)*(1+c*x))
def cdf_quad(x,c): return x*mp.quad(lambda u: pdf(x*u,c), [0,1])  # rescaled to avoid tiny-interval node issues
def ppf_root(q,c):
    q=mp.mpf(q); c=mp.mpf(c)
    # solve cdf(x)=q, cdf by quadrature; start from q (cdf~x for tiny c)
    x0 = q*mp.log1p(c)/c
    return mp.findroot(lambda x: cdf_quad(x,c)/q-1, x0, tol=mp.mpf(10)**(-mp.mp.dps+10))
cdfcases=[(1e-300,1e-300),(1e-30,1e-300),(1e-20,1e-300),(1e-16,1e-300),(1e-15,1e-300),(1e-14,1e-300),(1e-12,1e-300),(1e-300,1e-10)]
ppfcases=[(1e-300,1e-300),(1.2115276586286942e-26,1e-300),(1e-20,1e-300),(1e-16,1e-300),(1e-14,1e-300),(1e-12,1e-300),(1e-300,1e-10)]
def ulp(s,r): return float(abs(mp.mpf(s)-r)/(abs(r)*mp.mpf(2)**-52))
print("scipy",scipy.__version__)
for name,cases,ref,sf in [("cdf",cdfcases,cdf_quad,st.bradford.cdf),("ppf",ppfcases,ppf_root,st.bradford.ppf)]:
    for a,c in cases:
        vals=[]
        for d in (100,200):
            mp.mp.dps=d; vals.append(ref(mp.mpf(a),mp.mpf(c)))
        mp.mp.dps=200
        agree = abs(vals[0]-vals[1])<=abs(vals[1])*mp.mpf(10)**-30
        r=vals[1]
        extra=""
        if name=="ppf":
            extra=" cdf(xref)/q-1=%s"%mp.nstr(cdf_quad(r,mp.mpf(c))/mp.mpf(a)-1,3)
        s=sf(a,c)
        print(name,repr(a),repr(c),"scipy=",repr(s),"ref=",mp.nstr(r,20),"agree30=",agree,"ulp=%.3g"%ulp(s,r),extra)
