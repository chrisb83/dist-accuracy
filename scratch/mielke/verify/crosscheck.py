import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/mielke/verify")
from mpmath import mp
import ref_check as R

mp.dps = 60
print("--- cdf closed form vs mp.quad of the pdf ---")
for (x,k,s) in [(11.0,300.0,4.6),(2.0,1000.0,1.0),(1e10,1e6,1.0),
                (3.0,10.4,4.6),(0.5,2.0,3.0),(1.001,1e-3,1e6)]:
    c = R.cdf(x,k,s)
    # independent: integrate pdf in u=log t from -inf-ish to log x
    xm,km,sm = mp.mpf(x),mp.mpf(k),mp.mpf(s)
    f = lambda u: R.pdf(mp.e**u, k, s)*mp.e**u
    lo = mp.log(xm) - 200/km*mp.log(mp.mpf(10)) - 5
    q = mp.quad(f, [lo, mp.log(xm)-mp.mpf(1)/km, mp.log(xm)])
    print(f"x={x:<24g} k={k:<8g} s={s:<8g} closed={mp.nstr(c,20):24} quad={mp.nstr(q,20):24} relerr={mp.nstr(abs(c-q)/abs(c),3)}")

print("--- pdf as derivative of cdf (numerical diff) ---")
for (x,k,s) in [(11.0,300.0,4.6),(1e10,1e6,1.0),(3.0,10.4,4.6),(1.31830766936289e-309,1e-3,1e-3)]:
    d = mp.diff(lambda t: R.cdf(t,k,s), mp.mpf(x))
    p = R.pdf(x,k,s)
    print(f"x={x:<24g} k={k:<8g} s={s:<8g} pdf={mp.nstr(p,20):24} dcdf={mp.nstr(d,20):24} relerr={mp.nstr(abs(p-d)/abs(p),3)}")

print("--- ppf: cdf(x_ref) vs q ---")
for (q,k,s) in [(0.9999999999999999,10.4,4.6),(0.9999999999522942,1e6,1.0),
                (0.9999999999,1e6,1.0),(0.9999999999,1000.0,1.0),(0.85,1e6,1.0),
                (0.99999280314327,300.0,4.6),(0.9991516571017559,1000.0,1.0),
                (1e-300,2.0,3.0),(0.5,1e-3,1e6),(3.162277660168379e-151,0.5,2.0),
                (1.2961424830573168e-81,0.5,2.0),(1.997134533661588e-216,2.0,3.0),
                (0.9999992550002775,1e-3,1e6)]:
    x = R.ppf(q,k,s)
    back = R.cdf(x,k,s)
    print(f"q={q:<24g} k={k:<8g} s={s:<8g} x={mp.nstr(x,20):24} cdf(x)-q rel={mp.nstr(abs(back-mp.mpf(q))/mp.mpf(q),3)}")

print("--- logpdf vs log(pdf) ---")
for (x,k,s) in [(1e70,10.4,4.6),(1.001,1e-3,1e6),(1.35242013341704e+154,0.5,2.0)]:
    lp = R.logpdf(x,k,s); l2 = mp.log(R.pdf(x,k,s))
    print(f"x={x:<24g} k={k:<8g} s={s:<8g} logpdf={mp.nstr(lp,20):24} log(pdf)={mp.nstr(l2,20):24} relerr={mp.nstr(abs(lp-l2)/abs(lp),3)}")
