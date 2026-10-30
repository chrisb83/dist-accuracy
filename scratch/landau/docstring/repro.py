import numpy as np
import scipy
from scipy import stats
import mpmath as mp

mp.mp.dps = 40
pi = mp.pi

def f_doc(x):
    """pdf as written in the landau docstring (Landau's original form)."""
    g = lambda t: mp.exp(-t*mp.log(t) - x*t) * mp.sin(pi*t)
    return mp.quad(g, list(range(0, 41)) + [mp.inf]) / pi

def f_boost(x):
    """pdf in the form used by Boost.Math / Wikipedia."""
    g = lambda t: mp.exp(-t) * mp.cos(x*t + 2/pi*t*mp.log(t))
    return mp.quad(g, mp.linspace(0, 60, 61) + [mp.inf]) / pi

print("numpy", np.__version__, "scipy", scipy.__version__, "mpmath", mp.__version__)
print(f"{'x':>4} {'landau.pdf(x)':>22} {'docstring f(x)':>22} {'Boost-form f(x)':>22}")
for x in [-1, 0, 1, 3, 10]:
    print(f"{x:4} {stats.landau.pdf(x):22.16e} {mp.nstr(f_doc(x), 16):>22} {mp.nstr(f_boost(x), 16):>22}")

# Landau's original density is landau(loc=log(pi/2), scale=pi/2)
for lam in [-1, 0, 2]:
    print(f"lambda={lam:2}: f_doc={mp.nstr(f_doc(lam), 16)}  "
          f"landau.pdf(lam, loc=log(pi/2), scale=pi/2)={stats.landau.pdf(lam, np.log(np.pi/2), np.pi/2):.16e}")

# right tail: x*sf(x) -> 2/pi = 0.63662 for SciPy (it would be -> 1 for the docstring form)
for x in [1e3, 1e6]:
    print(f"x={x:.0e}: x*landau.sf(x) = {x*stats.landau.sf(x):.6f}")
