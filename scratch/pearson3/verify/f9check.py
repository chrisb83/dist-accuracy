import mpmath as mp, vref
for d in (100, 200, 300):
    mp.mp.dps = d
    # complementary-function route: Q(alpha, t), t = beta*(x - zeta), skew=-0.1 exact double
    a, b, z = vref.params(-0.1); t = b*(mp.mpf(-37.0) - z)
    g = mp.gammainc(a, t, mp.inf, regularized=True)
    # quad with more subdivision
    q = mp.quad(lambda u: vref.pdf(u, -0.1), mp.linspace(-37.0 - 80, -37.0, 81))
    print(d, mp.nstr(g, 25), mp.nstr(q, 25))
