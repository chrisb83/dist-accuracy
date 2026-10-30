# pdf with integer shapes: 2 ** (a + b - 1) is evaluated in int64
import warnings; warnings.simplefilter("ignore")
from scipy import stats
d = stats.jf_skew_t
for a, b in [(30, 33), (32, 32), (31, 33), (40, 40), (100, 3)]:
    print(a, b, d.pdf(0.5, a, b), d.pdf(0.5, float(a), float(b)))
