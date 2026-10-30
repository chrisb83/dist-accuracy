# gumbel_r hypotheses
- pdf = exp(logpdf): error ~ |logpdf| ulp, cond ~ |x(1-e^-x)| -> should be within cond (no pattern).
- logpdf = -x - exp(-x): no cancellation possible (x + e^-x >= 1). exp(-x) overflow for x < -709.78 -> -inf, true value below -1.8e308 (not normal). expect fine.
- cdf/logcdf/sf: closed forms with expm1, expect fine. sf for x < -3.6: true 1 - tiny, fine.
- ppf = -log(-log q): near q = 1/e (ppf = 0) log(y), y ~ 1: cond ~ 1/|ppf| -> within cond? (pattern 8). q near 1: log(q) of exact q fine.
- isf = -log(-log1p(-p)): near p = 1 - 1/e same; p near 1 fine.
- logsf inherited: log(sf) -> 0 for x < ~ -3.6 (sf rounds to 1), -inf for x > ~745 (pattern 12).
