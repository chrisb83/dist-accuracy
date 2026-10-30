# invgamma hypotheses (code lines 4965-4982)

y = 1/x throughout. cdf = Q(a, y), sf = P(a, y).

1. logpdf (4970) `-(a+1)*log(x) - gammaln(a) - 1/x`: pattern 5. Large a near the
   mode x ~ 1/(a+1): terms ~ a log a cancel to O(log a). Expect ~a log a / |result| ulp.
   Also near the zero crossing of logpdf (relative error unbounded).
2. pdf (4967) `exp(_logpdf)`: absolute error of logpdf -> relative error of pdf
   (gamma F2 analogue). Large a in the bulk; also far tails where |logpdf| ~ 700
   (but cond(x) ~ a+1 or 1/x there, maybe within conditioning).
3. cdf/sf (4973/4979): gamma F1 (igam_series 2000-term cap) for a >~ 2e5 and
   4.5 sqrt(a) < a - 1/x < 15 sqrt(a), i.e. x slightly above 1/a. gamma F6 prefactor
   for a = 1e3 with y = 500 / 2000.
4. ppf/isf (4976/4982): gammainccinv / gammaincinv inherit gamma F3 (large a) and
   F5 (a << 1, exponent/a). For a = 1e-3, isf/ppf at bulk q.
5. 1/x rounding: covered by cond(arg). 1/gammainc*inv(...) overflow only when the
   result is beyond the double range.
6. Inherited logcdf/logsf: cdf underflows for x -> 0 (Q(a, y) ~ y^(a-1) e^-y/Gamma(a)),
   sf underflows for x -> inf (P ~ y^a / Gamma(a+1)), e.g. a = 4, x > 1e77.
