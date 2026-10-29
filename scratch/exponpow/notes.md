# exponpow hypotheses
1. logpdf (l.2395): `1 + xb - exp(xb)` cancels for small xb (~ -xb^2/2, abs err eps). Matters when
   log(b) + (b-1)log(x) is also ~0, i.e. b = 1 (or b near 1) and small x. Pattern 5/2. Expect loss at b=1, x=1e-10.
   More generally: logpdf with b near 1 and x small: logpdf ~ log b + (b-1) log x - xb^2/2.
2. pdf = exp(logpdf) (l.2391): absolute error of a large |logpdf| (b<1, x tiny; or b>1 x tiny) becomes relative
   error ~|logpdf| ulp while cond(arg) ~ |b-1|. Pattern 3/4-ish (exp of log amplification). Expect minor/moderate.
3. ppf (l.2408) q near 1: log1p(-q) fine since q is exact; ok. Small b: pow(.,1/b) - cond also 1/b. Expect fine.
4. isf (l.2405): log1p(-log(q)): q near 1 -> log(q) ok. Expect fine.
5. cdf/sf: expm1 compositions, expect fine. sf large xb: cond large, fine.
6. logcdf = log(cdf) inherited: fine (cdf never cancels; cdf->1 log of cdf near 1 loses: log(cdf) where 1-cdf tiny -> log1p(-sf) better).
   logsf = log(sf) inherited: sf underflows when e^{xb} > ~745: logsf = 1 - exp(xb) = -expm1(xb) closed form.
