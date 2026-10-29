# genpareto hypotheses (SciPy 1.18.1, lines 3168-3282)

- H1 (pattern 9/10, cdf l.3227): -inv_boxcox1p(-x,-c). sc.inv_boxcox1p returns x when |lmbda*x| < 1e-154
  -> cdf = x for |c| tiny, any x. Probe: cdf(1, 1e-160) = 1.0 (true 0.632). Shared: scipy.special.inv_boxcox1p.
- H2 (pattern 3, logpdf l.3223 / logsf l.3234): c*x overflows for c*x > 1.8e308 (c > 1, x near 1e300) -> -inf.
  Probe: logsf(1e300, 1e10) = -inf. pdf = exp(logpdf) also -> 0? pdf true underflows there though.
- H3 (pattern 4/9, logpdf, logsf): tiny |c|: c*x subnormal or 0 -> log1p(cx)/c loses digits; -0 when cx underflows.
- H4 (pattern 3, isf l.3241, ppf l.3238): expm1(-c*log q) overflows while expm1(..)/c is finite;
  window c*L in (709.78, 709.78 + log c), L = -log q (isf) or -log1p(-q) (ppf). Large c (1e3, 1e6).
- H5 (pattern 6-ish, pdf l.3219): pdf = exp(logpdf) amplifies abs error of logpdf by |logpdf|; for c ~ 1, x huge,
  |logpdf| ~ 700 -> up to ~700 ulp; cond(x) ~ (1+c)/c small -> minor/moderate.
- H6 (pattern 7): apply_where c != 0 branch; c == 0 exactly fine; c = +-1e-8 check.
- H7 (c < 0): near upper endpoint -1/c, 1 + c x cancels; expected within conditioning.
- H8 boxcox/boxcox1p small-lambda threshold 1e-19: error ~ c*log q relative -> fine.
