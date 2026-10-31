import numpy as np, cmath
T = np.finfo(float).max/(1+2**0.5)
print("DBL_MAX/(1+sqrt2) =", T)
for z in [complex(-T*(1-1e-15), 0), complex(-T*(1+1e-15), 0), complex(0, T*(1-1e-15)), complex(0, T*(1+1e-15)),
          complex(1, T*1.01), complex(T*1.01, 3e307), complex(-T*1.01, 3e307), complex(-1e307, T*1.01)]:
    n = complex(np.sqrt(np.complex128(z))); c = cmath.sqrt(z)
    print(z, "re ratio", n.real/c.real if c.real else (n.real, c.real), "im ratio", n.imag/c.imag if c.imag else (n.imag, c.imag))
