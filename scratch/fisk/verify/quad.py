import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/fisk/verify')
from mpmath import mp
import ref
mp.dps = 60
for x, c in [(9.0, 3.085754862225318), (1.5, 100.0), (0.5, 1.2), (2.0, 0.5), (1e10, 1.2)]:
    cq = ref.cdf_quad(x, c); ca = ref.cdf(x, c)
    # sf via quad of upper tail
    print(f"x={x} c={c}: cdf_closed={mp.nstr(ca,30)} cdf_quad={mp.nstr(cq,30)} rel={mp.nstr(abs(ca-cq)/ca,3)}")
    print(f"   sf_closed={mp.nstr(ref.sf(x,c),30)}  1-cdf_quad={mp.nstr(1-cq,30)}")
# ppf inversion check
for q, c in [(1e-300, 3.085754862225318), (0.3, 1.2)]:
    xp = ref.ppf(q, c)
    print(f"ppf({q},{c})={mp.nstr(xp,25)} cdf(ppf)={mp.nstr(ref.cdf(xp,c),25)}")
