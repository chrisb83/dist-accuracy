# Verifier reference for k = 3 using the closed form of the range density of 3 normals:
# r_3(w) = 6 int phi(z) phi(z+w) [Phi(z+w)-Phi(z)] dz = (3/sqrt(pi)) exp(-w^2/4) erf(w/(2 sqrt 3))
# (complete the square: phi(z)phi(z+w) = e^{-w^2/4} e^{-y^2}/(2 pi), y = z + w/2, and
#  int e^{-y^2} Phi(y+a) dy = sqrt(pi) Phi(a/sqrt(3/2))).
# pdf(x) = int s f_S(s) r_3(x s) ds (1-D), cdf(x) = int f_S(s) R_3(x s) ds, R_3(w) = int_0^w r_3.
import mpmath as mp
from k2_ref import logfS

def r3(w):
    return 3/mp.sqrt(mp.pi)*mp.exp(-w*w/4)*mp.erf(w/(2*mp.sqrt(3)))

def R3(w):
    return mp.quad(r3, [0, w/2, w] if w < 10 else [0, 2, 5, 10, 20, mp.inf if w > 40 else w])

def ub(x):
    c = -mp.log(x)
    return [min(c-40, -40), -20, -10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]

def pdf3(x, df):
    x, df = mp.mpf(x), mp.mpf(df)
    return mp.quad(lambda u: mp.exp(logfS(mp.exp(u), df) + 2*u)*r3(x*mp.exp(u)), ub(x))

def cdf3(x, df):
    x, df = mp.mpf(x), mp.mpf(df)
    return mp.quad(lambda u: mp.exp(logfS(mp.exp(u), df) + u)*R3(x*mp.exp(u)), ub(x))
