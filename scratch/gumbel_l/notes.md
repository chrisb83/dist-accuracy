# gumbel_l hypotheses
- pdf = exp(logpdf) (line 4467): error of logpdf (~|logpdf|*eps abs) amplified; for x>0 logpdf ~ -e^x, error ~ e^x ulp; cond(x) = |x(1-e^x)| ~ x e^x, so likely within conditioning. Test x in [1, 6.5] and x << 0.
- logpdf x - exp(x): no cancellation (e^x - x >= 1). Expect fine.
- cdf -expm1(-exp(x)): fine; left tail cdf ~ e^x underflows at x < -745 (subnormal, not measured).
- sf exp(-exp(x)): fine.
- logsf -exp(x): fine.
- ppf log(-log1p(-q)): near q = 1-1/e (ppf=0), log(y) y~1 -> cond large; check error vs cond. q -> 1: fine since 1-q exact-ish.
- isf log(-log(q)): q near 1: log(q) of exact q is accurate. q = 1/e: cond.  q tiny: fine.
- logcdf inherited log(cdf): -inf for x < -745 (cdf underflows), loses for subnormal cdf. Better: x - e^x/2... i.e. log(-expm1(-t)) with t=e^x; for small t: x + log1p(-t/2+...) -> x + log(-expm1(-t)/t).
