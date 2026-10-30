# Check that the F4/F5/F7 errors sit in sc.hyp2f1 / sc.beta (mechanism check).
import mpmath as mp, scipy.special as sc
mp.mp.dps = 60
def ulp(s, r): return float(abs(mp.mpf(s)-r)/abs(r)/mp.mpf(2)**-52)
for c,a,b,z in [(2.500000000001,2.5,1.5,1e6),(2.5,2.5,1.5,1e10),(1e4,3,2,1e-4)]:
    # Euler transform 2F1(c,a;a+b;-z) = (1+z)^(-a) 2F1(a, a+b-c; a+b; z/(1+z))  (different route)
    r = (1+mp.mpf(z))**(-mp.mpf(a))*mp.hyp2f1(a, a+b-mp.mpf(c), a+b, mp.mpf(z)/(1+mp.mpf(z)))
    r2 = mp.hyp2f1(c,a,a+b,-mp.mpf(z))
    print('hyp2f1', c,a,b,z, 'ulp', ulp(sc.hyp2f1(c,a,a+b,-z), r), 'routes agree', mp.nstr(r/r2-1,3))
for a,b in [(2,1e6),(2,1e4)]:
    print('beta', a,b,'ulp', ulp(sc.beta(a,b), mp.beta(a,b)))
