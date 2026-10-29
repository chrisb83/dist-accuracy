# F2 re-check (verifier disputed mechanism)
- ndtr_tail.py / ndtr_tail.txt: sc.ndtr alone vs mpmath, x in [-38,-5]. Errors 0.1 ... 514 ulp,
  power-of-two clustered (511/256/128/64/32/16): about one ulp of the internal exponent z^2 ~ x^2/2.
  Within ndtr's own cond (~x^2, 1.37e3 at -37). erfcx and log_ndtr are <= 1.8 ulp.
- ndtr_mech.py: not just the rounding of z = x/sqrt2; it is the rounding of z^2 inside erfc's exp(-z^2).
- decompose.py / decompose.txt: per-term absolute error divided by F (in ulp of F).
  erfcx branch (K<1): term2 accurate at integer x (0.1-0.4 ulp), F2 there is ~entirely ndtr * Phi/F.
    Non-integer x: term2 picks up rounding of 0.5*x**2 (26 ulp at x=-30.3), so both terms contribute.
  default branch (K>=1): exp(logprod) is 100-700 ulp off itself, comparable to ndtr; at x=-33,-28 (K=1.5)
    ndtr is accurate and the whole 1.6e4-1.8e4 ulp comes from exp(logprod). At -37, K=1.5 the ndtr part
    is 2.9e4 and the term2 part 6.8e3 F-ulp (partially cancelling to 2.21e4).
