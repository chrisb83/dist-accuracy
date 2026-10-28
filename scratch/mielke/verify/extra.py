import sys, warnings
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/mielke/verify")
from mpmath import mp
import ref_check as R
from scipy.stats import mielke
warnings.simplefilter("ignore")
EPS=2.220446049250313e-16
cases=[("F7","logpdf",1e290,1e6,1.0),("F7","logpdf",1.8087545456327052e+16,1000.,1.0),
       ("F8","ppf",1.6155980984399093e-213,1.0,1e-3),("F9","cdf",8.0,1.0,1e-3),
       ("F9","cdf",1.399787605293446e+102,2.0,3.0)]
for fid,meth,a,k,s in cases:
    sv=float(getattr(mielke,meth)(a,k,s))
    vals=[]
    for d in (100,200):
        mp.dps=d; vals.append(getattr(R,meth)(a,k,s))
    mp.dps=60
    rel=abs(vals[0]-vals[1])/abs(vals[1])
    u=float(abs(mp.mpf(sv)-vals[1])/(abs(vals[1])*mp.mpf(EPS)))
    mp.dps=25
    print(f"{fid} {meth}({a:g},{k:g},{s:g}): scipy={sv!r} ref={mp.nstr(vals[1],17)} ulp={u:.3g} stab={mp.nstr(rel,2)}")
