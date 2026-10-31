# k = 3 via Owen's T: R_3(w) = 1 - 12 T(w/sqrt2, 1/sqrt3); averaging exp(-c S^2) over
# S^2 ~ chi2_df/df gives (1 + 2c/df)^(-df/2), so
#   cdf(x) = 1 - (6/pi) int_0^{1/sqrt3} (1+y^2)^-1 (1 + x^2 (1+y^2)/(2 df))^(-df/2) dy
#   pdf(x) = (3x/pi)  int_0^{1/sqrt3} (1 + x^2 (1+y^2)/(2 df))^(-df/2-1) dy
# For small x the cdf is also computed as int_0^x pdf (no cancellation).
import mpmath as mp
def pdfO(x, df):
    x, df = mp.mpf(x), mp.mpf(df)
    return 3*x/mp.pi*mp.quad(lambda y: (1 + x*x*(1+y*y)/(2*df))**(-df/2-1), [0, 1/mp.sqrt(3)])
def cdfO(x, df):
    x, df = mp.mpf(x), mp.mpf(df)
    return 1 - 6/mp.pi*mp.quad(lambda y: (1 + x*x*(1+y*y)/(2*df))**(-df/2)/(1+y*y), [0, 1/mp.sqrt(3)])
def cdfP(x, df):
    return mp.quad(lambda t: pdfO(t, df), [0, mp.mpf(x)])
