# Is sc.ndtr's tail error just the rounding of z = x/sqrt(2) (input perturbation)?
import numpy as np, scipy.special as sc, mpmath as mp
mp.mp.dps = 60; eps = 2.220446049250313e-16
for x in [-37.0,-36.0,-35.0,-33.0,-30.0,-30.3,-20.7,-10.0]:
    X=mp.mpf(x); r=mp.erfc(-X/mp.sqrt(2))/2
    z = x*0.70710678118654752440  # NPY_SQRT1_2 product, rounded
    rz = mp.erfc(-mp.mpf(z))/2    # exact erfc at the rounded z
    print(x, "ndtr ulp", round(float(abs(mp.mpf(sc.ndtr(x))-r)/r/eps),2),
          "| exact erfc(fl(z)) vs true", round(float(abs(rz-r)/r/eps),2),
          "| ndtr vs exact erfc(fl(z))", round(float(abs(mp.mpf(sc.ndtr(x))-rz)/rz/eps),2))
