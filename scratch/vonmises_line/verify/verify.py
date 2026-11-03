import mpmath as mp, numpy as np, scipy, scipy.stats as st
pts = [(-2.5,20),(-3.0,20),(-1.2566370614359172,49.9),(-2.827433388230814,49.9),
       (-1.0,49.9),(-1.5,20),(-2.0,10),(-1.0,20),(-3.0,3.99390425810714),(-2.0,3.99390425810714),
       (-3.0,200),(-3.0,50),(-1.0,50),(-1.0,1000),(-0.5,200),(-0.1,50),(0.05,50),(-0.01,50)]
# Route A: direct quad of docstring pdf exp(k cos t)/(2 pi I0(k)), written in scaled form
def cdf_quad(x,k):
    x=mp.mpf(x);k=mp.mpf(k)
    f=lambda t: mp.exp(k*(mp.cos(t)-1))
    a=-mp.pi
    nodes=[a]+[a+(x-a)*mp.mpf(i)/16 for i in range(1,16)]+[x]
    return mp.quad(f,nodes)/(2*mp.pi*mp.besseli(0,k)*mp.exp(-k))
# Route B: Fourier series F = (x+pi)/(2pi) + (1/pi) sum I_j(k)/I_0(k) sin(jx)/j  (needs high dps)
def cdf_series(x,k):
    x=mp.mpf(x);k=mp.mpf(k);i0=mp.besseli(0,k)
    s=(x+mp.pi)/(2*mp.pi); j=1
    while True:
        r=mp.besseli(j,k)/i0
        s+=r*mp.sin(j*x)/(j*mp.pi)
        if r<mp.mpf(10)**(-mp.mp.dps-5) and j>k: break
        j+=1
    return s
print("scipy",scipy.__version__)
for x,k in pts:
    mp.mp.dps=100; a1=cdf_quad(x,k)
    mp.mp.dps=200; a2=cdf_quad(x,k)
    mp.mp.dps=450; b=cdf_series(x,k)
    agree=abs(a1-a2)/a2; ab=abs(a2-b)/a2
    sp=st.vonmises_line.cdf(x,k)
    ref=float(a2)
    rel=abs(sp-ref)/ref
    print(f"cdf({x!r},{k!r}) scipy={sp!r} ref={mp.nstr(a2,17)} quad100v200={mp.nstr(agree,3)} quad_v_series={mp.nstr(ab,3)} rel={rel:.3g} ulp={rel/2.220446049250313e-16:.3g}")
