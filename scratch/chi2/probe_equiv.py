import numpy as np, scipy.special as sc
rng = np.random.default_rng(0)
for k in [2e-3, 1.0, 2.0, 55.0, 2e3, 2e6, 2e8]:
    x = k * np.exp(rng.uniform(-3, 3, 2000)); x = np.concatenate([x, k + np.sqrt(2*k)*np.linspace(-30, 30, 601)]); x = x[x > 0]
    q = 10.0 ** rng.uniform(-300, 0, 2000)
    d1 = np.sum(sc.chdtr(k, x) != sc.gammainc(k/2, x/2))
    d2 = np.sum(sc.chdtrc(k, x) != sc.gammaincc(k/2, x/2))
    d3 = np.sum(~np.isclose(sc.chdtri(k, q), 2*sc.gammainccinv(k/2, q), rtol=0, atol=0) & ~(sc.chdtri(k, q) == 2*sc.gammainccinv(k/2, q)))
    print(k, "chdtr!=gammainc:", d1, " chdtrc!=gammaincc:", d2, " chdtri!=2*gammainccinv:", d3)
