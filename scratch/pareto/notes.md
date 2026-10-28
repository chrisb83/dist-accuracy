# pareto hypotheses (SciPy 1.18.1, lines 8236-8250)

1. cdf `1 - x**(-b)` (l.8241): pattern 1/2, cancellation near x=1 (left tail, cdf small) and for small b at any moderate x. Expected loss: x = 1+1e-15 with b=2.6 -> cdf ~ 2.6e-15, rel err ~ ...; small b (1e-6) -> cdf ~ b*log(x), large rel error. Fix -expm1(-b*log(x)).
2. ppf `pow(1-q, -1/b)` (l.8244): pattern 8, 1-q rounds for q < eps -> returns 1.0 exactly (true 1 + q/b...). Loss of relative info only in (x-1); relative error of x itself is ~q/b which may be small vs 1. With small b, q/b large -> visible. e.g. b=1e-10, q=1e-12: true x = exp(1e-2)... measurable.
3. pdf `b * x**(-b-1)` (l.8238): pattern 4, x**(-b-1) subnormal while b large keeps result normal (b=1e6? x**(-1e6-1) needs pdf ~ b*tiny; need b ~ 1e10+ to lift by many orders). Also x**(-b-1) with b tiny: -b-1 rounds to -1 when b<1e-16 -> pdf = b/x * x**(-b) loses factor x**(-b); for huge x (1e300) with b=1e-17: x**-b = exp(-6.9e-15) ~ 1-6.9e-15 -> ~31 ulp. Minor.
4. isf `q**(-1/b)` and sf `x**(-b)`: direct pow, expected fine (cond ~ |b log x|).
5. Inherited logpdf/logcdf/logsf: pdf/sf underflow for large x or large b -> -inf; logcdf near x=1 inherits F1.
