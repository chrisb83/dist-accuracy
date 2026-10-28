import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/mielke/verify")
from mpmath import mp
import ref_check as R
mp.dps = 50
# integrate pdf in u = log t with panels chosen around the mode of t*pdf
for (x,k,s) in [(2.0,1000.0,1.0),(1e10,1e6,1.0),(1.001,1e-3,1e6)]:
    xm,km,sm = mp.mpf(x),mp.mpf(k),mp.mpf(s)
    f = lambda u: R.pdf(mp.e**u, k, s)*mp.e**u
    ux = mp.log(xm)
    # mass of t*pdf decays like e^{k u} below the bulk: width ~ 1/k
    pts = [ux - 300*mp.log(mp.mpf(10))]
    for d in [200,100,50,20,10,5,2,1,mp.mpf(1)/2,mp.mpf(1)/10,mp.mpf(1)/100,
              mp.mpf(1)/km*10, mp.mpf(1)/km, 0]:
        v = ux - d
        if v > pts[-1]:
            pts.append(v)
    pts = sorted(set(pts))
    q = mp.quad(f, pts)
    c = R.cdf(x,k,s)
    print(f"x={x:<12g} k={k:<8g} s={s:<8g} closed={mp.nstr(c,18):22} quad={mp.nstr(q,18):22} rel={mp.nstr(abs(c-q)/abs(c),3)}")
