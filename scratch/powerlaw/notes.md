# powerlaw hypotheses
- pdf `a*x**(a-1.0)`: a-1.0 inexact for a<0.5 -> rounded exponent * |log x| (pattern 4 / INDEX 4), minor, small x, small a.
- pdf: large a, x**(a-1) subnormal, a*x**(a-1) normal (pattern 4), a=1e6.
- logpdf `log(a)+xlogy(a-1,x)`: cancellation near pdf=1 (pattern 5) - but cond(x) large there.
- ppf `pow(q, 1/a)`: 1/a rounding * |log q| (INDEX 4), minor, small q, a=3.
- sf `-powm1(x,a)`: check powm1 accuracy for tiny a, large a; overflow impossible (x<=1, a>0).
- inherited isf = ppf(1-q): rounding of 1-q amplified by 1/a for small a (pattern 8/INDEX 3).
- inherited logsf = log(sf): sf subnormal for tiny a near x=1.
