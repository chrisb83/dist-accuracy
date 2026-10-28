# gamma hypotheses (code reading, SciPy 1.18.1, lines 3581-3788)

_pdf   = exp(_logpdf)
_logpdf= sc.xlogy(a-1, x) - x - sc.gammaln(a)
_cdf   = sc.gammainc(a, x)
_sf    = sc.gammaincc(a, x)
_ppf   = sc.gammaincinv(a, q)
_isf   = sc.gammainccinv(a, q)

H1 (pattern 5, logpdf/pdf): for large a and x ~ a, (a-1)*log(x) ~ a*log a and
   gammaln(a) ~ a*log a - a are both ~1e7 for a=1e6 while logpdf ~ -8.
   Absolute error ~ a*log(a)*eps -> logpdf rel err ~ 1e-10 (6e5 ulp) and
   pdf rel err ~ a*log(a)*eps (~1e6-1e7 ulp). Regime: a >= 1e3, x near a.
H2 (pattern 5/3, pdf = exp(logpdf)): pdf amplifies the *absolute* error of
   logpdf; wherever |logpdf| is large (large x, large a) pdf loses
   |logpdf|*eps relative. x=1e300: logpdf ~ -1e300 -> pdf underflows (no
   finding), but x=700, a=2: logpdf ~ -694 -> ~700*eps = 3e2 ulp.
H3 (pattern 10, cdf/sf): gammainc/gammaincc themselves. Check large a near
   x = a (uniform asymptotic expansion) and tiny a (cdf ~ x**a/Gamma(a+1)).
H4 (pattern 8/10, ppf): gammaincinv for q near 1 (ill conditioned, but check
   beyond conditioning), for tiny a (ppf = (q*Gamma(a+1))**(1/a), a rounding
   error in the exponent 1/a is amplified by 1/a), and for large a.
H5 (pattern 8/10, isf): gammainccinv for tiny q and large a.
H6 (pattern 12, logcdf/logsf inherited): log(cdf)/log(sf) -> -inf once
   cdf/sf underflow; true logcdf/logsf are finite far out
   (logsf ~ -x + (a-1)log x - gammaln(a)).
H7 (pattern 2, logcdf near x -> inf): log(cdf) = log1p(-sf) loses digits
   for sf < 1e-16? inherited, opportunity only.
