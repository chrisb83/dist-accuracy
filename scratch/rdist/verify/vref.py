# Independent verifier reference for rdist (standard form).
# pdf: (1-x^2)^(c/2-1)/B(1/2,c/2) directly.
# cdf: via X^2 ~ Beta(1/2, c/2): F(x) = 1/2 + sign(x)/2 * I_{x^2}(1/2, c/2);
#      tails: F(x<0) = 1/2 * I_{1-x^2}(c/2, 1/2) (no subtraction).
# ppf: secant/Newton on the same identity, checked by cdf(x_ref) == q.
import mpmath as mp

def M(v): return mp.mpf(float(v))

def pdf(x, c):
    x, c = M(x), M(c)
    return (1 - x*x)**(c/2 - 1) / mp.beta(mp.mpf(1)/2, c/2)

def logpdf(x, c): return mp.log(pdf(x, c))

def cdf(x, c):
    x, c = M(x), M(c)
    if x < 0:
        return mp.betainc(c/2, mp.mpf(1)/2, 0, 1 - x*x, regularized=True)/2
    return 1 - mp.betainc(c/2, mp.mpf(1)/2, 0, 1 - x*x, regularized=True)/2

def sf(x, c): return cdf(-M(x), c)

def ppf(q, c):
    q, c = M(q), M(c)
    t = 2*q - 1  # exact in mpmath
    s = 1 if t > 0 else -1
    t = abs(t)
    b = c/2
    # |x| solves I_{x^2}(1/2, b) = t ; solve in u = log|x|
    import scipy.special as sc, math
    g = math.sqrt(float(sc.betaincinv(0.5, float(b), float(t))))
    G = lambda u: mp.log(mp.betainc(mp.mpf(1)/2, b, 0, mp.exp(2*u), regularized=True)) - mp.log(t)
    u = mp.findroot(G, (mp.log(g), mp.log(g)*(1+mp.mpf(10)**-8)), solver='secant', tol=mp.mpf(10)**(-2*mp.mp.dps+20), maxsteps=200)
    return s*mp.exp(u)

def quadcdf(x, c):
    x, c = M(x), M(c)
    return mp.quad(lambda t: (1-t*t)**(c/2-1), [-1, 0, x]) / mp.beta(mp.mpf(1)/2, c/2) if x > 0 else \
           mp.quad(lambda t: (1-t*t)**(c/2-1), [-1, x]) / mp.beta(mp.mpf(1)/2, c/2)
