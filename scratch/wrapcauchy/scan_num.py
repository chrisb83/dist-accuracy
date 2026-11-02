# relative error of 1.0 - c*c (pdf numerator, line 11480) vs exact rational value
from fractions import Fraction as F
import numpy as np
for k in [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 3e-16]:
    worst = (0, None)
    for c in np.linspace(1 - 2 * k, 1 - k, 200):
        num = 1.0 - c * c
        ex = 1 - F(c) ** 2
        r = abs(float((F(num) - ex) / ex)) / 2.220446049250313e-16
        if r > worst[0]:
            worst = (r, c)
    print(f"1-c ~ {k:g}: worst {worst[0]:.3g} ulp at c={worst[1]!r}")
