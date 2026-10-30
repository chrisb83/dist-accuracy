"""Error of individual pdf factors (pattern 6) and hyp2f1 near-degenerate cases."""
import mpmath as mp, numpy as np, scipy.special as sc
mp.mp.dps = 60
u = lambda s, r: float(abs(mp.mpf(s) / r - 1) / 2.22e-16)
print("(1-x)**(b-1), b=1e4:")
for x in [1.6136787423080556e-3, 6.24075628291297e-4, 1e-3, 0.0123, 0.3]:
    s = (1. - x) ** (1e4 - 1.0); r = (1 - mp.mpf(x)) ** (mp.mpf(1e4) - 1)
    print("  x", x, "ulp", round(u(s, r)), "cond", (1e4 - 1) * x / (1 - x))
print("(1+z*x)**c, c=1e4, z=1e-4:")
for x in [0.5, 0.15, 0.99, 1e-3]:
    s = (1.0 + 1e-4 * x) ** 1e4; r = (1 + mp.mpf(1e-4) * mp.mpf(x)) ** mp.mpf(1e4)
    print("  x", x, "ulp", round(u(s, r)), "cond", 1e4 * 1e-4 * x / (1 + 1e-4 * x))
print("(1+z*x)**c, c=1e6, z=1e-6 (large c, beta/hyp2f1 aside):")
for x in [0.3, 0.7]:
    s = (1.0 + 1e-6 * x) ** 1e6; r = (1 + mp.mpf(1e-6) * mp.mpf(x)) ** mp.mpf(1e6)
    print("  x", x, "ulp", round(u(s, r)))
print("x**(a-1), a=0.01:", u(1e-300 ** (0.01 - 1.), mp.mpf(1e-300) ** (mp.mpf(0.01) - 1)),
      " a-1 rounding:", float(mp.mpf(0.01 - 1.) - (mp.mpf(0.01) - 1)))
print("hyp2f1(c, a; a+b; -z), c near a, z = 1e6:")
for c in [2.5, 2.5 + 1e-12, 2.5 + 1e-8, 2.5 + 1e-4, 2.6, 3.5, 1.5]:
    s = sc.hyp2f1(c, 2.5, 4.0, -1e6); r = mp.hyp2f1(c, 2.5, 4, -1e6)
    print("  c", c, "ulp", f"{u(s, r):.3g}")
print("hyp2f1(c, a; a+b; -z), a == c, z sweep:")
for z in [10, 100, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12]:
    s = sc.hyp2f1(2.5, 2.5, 4.0, -z); r = mp.hyp2f1(2.5, 2.5, 4, -z)
    print("  z", z, "ulp", f"{u(s, r):.3g}")
print("hyp2f1(c, 3; 5; -z) large c:")
for c, z in [(30, 1e-2), (100, 1e-2), (100, 1e-3), (1e3, 1e-3), (1e3, 1e-4), (1e4, 1e-5), (1e4, 1e-3), (1e5, 1e-5)]:
    s = sc.hyp2f1(c, 3, 5, -z); r = mp.hyp2f1(c, 3, 5, -z)
    print("  c", c, "z", z, "ulp", f"{u(s, r):.3g}", repr(s))
