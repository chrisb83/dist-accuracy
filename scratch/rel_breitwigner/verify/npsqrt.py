import numpy as np, cmath, mpmath as mp
mp.mp.dps = 50
zs = [complex(-1e308, -1e154), complex(-7.5e307, 0), complex(-1e308, 0), complex(1e308, 1),
      complex(-7.4e307, 0), complex(-7.46e307, 0), complex(7.5e307, 0), complex(0, 1.5e308),
      complex(-1e308*1.0, -1e154)]
for z in zs:
    n = complex(np.sqrt(np.complex128(z)))
    c = cmath.sqrt(z)
    m = mp.sqrt(mp.mpc(z.real, z.imag))
    print(z, "np:", n, "cmath:", c, "mp:", mp.nstr(m, 17))
# scalar vs array, and threshold scan
print("array:", np.sqrt(np.array([complex(-1e308, 0)])))
lo, hi = 1e307, 1.7e308
for _ in range(80):
    mid = (lo*hi)**0.5
    ok = abs(np.sqrt(np.complex128(complex(-mid, 0))).imag - mid**0.5) < 1e-10*mid**0.5
    lo, hi = (mid, hi) if ok else (lo, mid)
print("threshold |Re| ~", lo, hi, "sqrt ->", lo**0.5)
# the argument SciPy forms for rho = 1e154 and 8.6e153 / 8.65e153
for rho in [8.6e153, 8.65e153, 1e154, 1.34e154]:
    w = -rho*(rho + 1j)
    print(rho, w, "np:", np.sqrt(w), "cmath:", cmath.sqrt(complex(w)))
print(np.__version__, np.show_runtime() if False else "")
