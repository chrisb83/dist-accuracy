import numpy as np, cmath, mpmath as mp, warnings
warnings.simplefilter('ignore')
mp.mp.dps = 50
print("numpy", np.__version__)
M = np.finfo(float).max
T = M/(1+2**0.5)
print("T = DBL_MAX/(1+sqrt2) =", repr(T))
def cmp(z):
    n = complex(np.sqrt(np.complex128(z)))
    m = mp.sqrt(mp.mpc(z.real, z.imag))
    mr, mi = float(m.real), float(m.imag)
    rr = n.real/mr if mr else (n.real, mr)
    ri = n.imag/mi if mi else (n.imag, mi)
    return n, complex(mr, mi), cmath.sqrt(z), rr, ri
cases = [complex(-1e308,-1e154), complex(-7.5e307,0), complex(-1e308,0), complex(1e308,1),
         complex(1e308,0), complex(0,1.5e308), complex(1,1.5e308), complex(-1e300,1.5e308),
         complex(1e300,-1.5e308), complex(7.4e307,1), complex(-7.4e307,-1)]
for z in cases:
    n, m, c, rr, ri = cmp(z)
    print(z, "np", n, "mp", m, "cmath ok", abs(c-m) <= 1e-15*abs(m), "re ratio", rr, "im ratio", ri)
# bisect threshold on |Re| (negative real axis, perturbation in imag) and on |Im| (pure imaginary)
def bad_re(a):  # z = -a - 1j
    _,_,_,rr,ri = cmp(complex(-a, -1.0)); return abs(ri-1) > 1e-12
def bad_re_pos(a):
    _,_,_,rr,ri = cmp(complex(a, 1.0)); return abs(ri-1) > 1e-12
def bad_im(a):
    _,_,_,rr,ri = cmp(complex(1.0, a)); return abs(ri-1) > 1e-12 or abs(rr-1)>1e-12
for name, f in [("Re<0", bad_re), ("Re>0", bad_re_pos), ("Im", bad_im)]:
    lo, hi = 1e307, 1.7e308
    assert not f(lo) and f(hi)
    while np.nextafter(lo, np.inf) < hi:
        mid = lo + (hi-lo)/2
        if mid in (lo, hi): break
        lo, hi = (mid, hi) if not f(mid) else (lo, mid)
    print(name, "last ok", repr(lo), "first bad", repr(hi), "hi==T?", hi == T, "lo<T<=hi", lo < T <= hi)
# which part is wrong, by how much, positive real
for z in [complex(1e308,1), complex(-1e308,1), complex(1,1e308), complex(-1,1e308), complex(1e308,1e300)]:
    n, m, c, rr, ri = cmp(z); print("parts", z, "re ratio", rr, "im ratio", ri)
