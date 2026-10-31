# skewcauchy hypotheses (SciPy 1.18.1, lines 9654-9718)

1. cdf, x <= 0 branch, line 9698: (1-a)/2 + (1-a)/pi*arctan(x/(1-a)). Left tail x -> -inf:
   arctan -> -pi/2, the two terms cancel (pattern 1/5). Loss for |x| >~ (1-a)/eps.
   Fix: (1-a)/pi * arctan2(1-a, -x)  (cauchy uses arctan2(1, -x)).
2. ppf, left branch, line 9705: tan(pi/(1-a)*(q - (1-a)/2))*(1-a). Small q: q - (1-a)/2
   loses q (pattern 8); tan at -pi/2 + O(eps). Loss for q <~ eps*(1-a).
   Fix: -(1-a)/tan(pi*q/(1-a)).
3. ppf, right branch, q -> 1: pole of tan, argument pi/(1+a)*(q-q0) rounded; probably within
   cond ~ 1/(1-q). Check, esp. a near -1 (1+a tiny).
4. ppf near median q0 = (1-a)/2: (1-a)/2 rounded when 1-a inexact (a in (-1,0.5)); cond
   large there anyway. Check.
5. pdf: x**2 overflow at |x|>1.34e154 -> 0; true pdf ~ (1+-a)^2/(pi x^2) subnormal there
   (1+-a <= 2), so not measurable. a near +-1: (1+-a) exact (Sterbenz) -> fine expected.
6. Inherited: sf = 1-cdf (right tail loss; closed form (1+a)/pi*arctan2(1+a, x)),
   isf = ppf(1-q) (small q loss; (1+a)/tan(pi q/(1+a))), logpdf = log(pdf) (pdf
   subnormal/0 for |x|>~1e154), logcdf/logsf = log of those.
