import numpy as np, scipy.stats as st
def series(k,x,p):
    s=np.sin(x);c=np.cos(x);sn=np.sin(p*x);cn=np.cos(p*x);R=0.;V=0.
    for n in range(p-1,0,-1):
        sn,cn=sn*c-cn*s,cn*c+sn*s
        R=1./(2*n/k+R); V=R*(sn/n+V)
    return 0.5+x/(2*np.pi)+V/np.pi
for x,k in [(-2.5,20),(-3.0,20),(-1.2566370614359172,49.9),(-2.827433388230814,49.9),(-1.0,49.9),(-1.5,20),(-2.0,10),(-1.0,20),(-3.0,3.99390425810714)]:
    p=int(np.ceil(28.5+0.5*k-100/(k+5))) if False else int(29+0.5*k-100/(k+5))
    r=series(k,x,p); r=min(max(r,0.),1.)
    print(x,k,p,repr(r),repr(st.vonmises_line.cdf(x,k)), r==st.vonmises_line.cdf(x,k))
