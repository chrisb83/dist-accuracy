# chi2 hypotheses (SciPy 1.18.1, lines 1608-1693); chi2(df) == gamma(a=df/2, scale=2)

_pdf    = exp(_logpdf)                                           (1650)
_logpdf = xlogy(df/2-1, x) - x/2 - gammaln(df/2) - log(2)*df/2   (1653)
_cdf    = chdtr(df, x)   = igam(df/2, x/2)                       (1656)
_sf     = chdtrc(df, x)  = igamc(df/2, x/2)                      (1659)
_isf    = chdtri(df, p)  = 2*igamci(df/2, p)                     (1662)
_ppf    = 2*gammaincinv(df/2, p)                                 (1665)

H1 (pattern 5, logpdf): large df, x ~ df: xlogy ~ (df/2) log df, gammaln(df/2), log2*df/2
   all ~ df log df / 2, result O(log df). gamma F4 analogue. df >= 2e3.
H2 (pattern 5/3, pdf = exp(logpdf)): pdf rel err ~ max term * eps. gamma F2 analogue,
   ~1e6-1e7 ulp at df = 2e6; ~500 ulp in far tails of moderate df.
H3 (pattern 10, cdf/sf): igam 2000-term series cap (gamma F1 / issue-reports/gammainc-large-a.md)
   predicted for df >~ 4e5, x in df - k*sqrt(2 df), 4.5 < k < ~15. Test df = 2e6 and 2e8 (aux ref).
H4 (pattern 10, ppf/isf): gammaincinv / igamci inherit H3 (gamma F3); small df (2e-3) exp(./a)
   amplification (gamma F5).
H5 (pattern 5, cdf/sf): igam_fac exp(a log x - x - lgam a) prefactor (gamma F6), df=2e3 x far from df.
H6 chdtri vs gammainccinv: same igamci? check isf at small q / tiny df.
H7 (pattern 12) inherited logcdf/logsf: log(sf) -> -inf once sf underflows (x >~ 1500 for df=55);
   logcdf -inf in far left tail for large df.
