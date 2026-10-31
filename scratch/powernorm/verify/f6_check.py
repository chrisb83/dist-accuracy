import mpmath as mp
for d in (100, 200):
    mp.mp.dps = d
    for x, c in ((-38, 1e10), (-38.5, 1e20)):
        x = mp.mpf(x); c = mp.mpf(c)
        Phi = mp.erfc(-x/mp.sqrt(2))/2
        cdf_series = c*Phi - c*(c-1)/2*Phi**2          # 1-(1-Phi)^c, binomial
        # quad of pdf with substitution t = x - u, integrand scaled
        pdf = lambda t: c*mp.npdf(t)*(1-mp.erfc(-t/mp.sqrt(2))/2)**(c-1)
        q = mp.quad(lambda u: pdf(x-u), [0, 0.01, 0.05, 0.2, 1, mp.inf])
        pdf_direct = c*mp.npdf(x)*mp.power(1-Phi, c-1)
        print(d, mp.nstr(x,4), mp.nstr(cdf_series,20), mp.nstr(q/cdf_series-1,3), 'pdf', mp.nstr(pdf_direct,20))
