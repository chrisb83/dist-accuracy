import sys; sys.path.insert(0, '.')
import mpmath as mp
from vref import pdf
mp.mp.dps = 40
U = 1/mp.mpf(1e308); S = U**mp.mpf(0.005)
g = lambda s: (lambda u: pdf(1/u,1,0.01,1)/u**2 * 200*s**199)(s**200)
print('F8 sf quad (u=s^200)', mp.nstr(mp.quad(g, [0, S]), 17))
