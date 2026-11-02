import mpmath as mp
mp.mp.dps=60
for xs in [1e-312, "1e-312"]:
    x=mp.mpf(xs); b=mp.mpf(1e-12)
    print(repr(xs), mp.nstr(x,20), mp.nstr(mp.expm1(-x)/mp.expm1(-b),20))
x=mp.mpf(1e-310); b=mp.mpf(1e-300)
print(mp.nstr(x,20), mp.nstr(mp.expm1(-x)/mp.expm1(-b),20))
mp.mp.dps=200
b=mp.mpf(20); print(mp.nstr(-mp.log(-mp.expm1(-b)),20))
