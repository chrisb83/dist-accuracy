import sys; sys.path.insert(0, '.')
import mpmath as mp
from vref import pdf_pois
mp.mp.dps = 50
x = mp.mpf('625.75169309670095'); q = mp.mpf(0.9999999999999627)
print(mp.nstr(q/(x*pdf_pois(x,10,300)), 5))
