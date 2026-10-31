# powernorm hypotheses (SciPy 1.18.1, lines 8980-9031)

Q(x) = ndtr(-x) (the sf of the normal). logsf = c*log_ndtr(-x) is the core.

H1 pdf (l.9013) `c*phi(x)*ndtr(-x)**(c-1)`, x > ~37.5 with c < 1: ndtr(-x) underflows
   to 0, 0**(c-1) = inf, phi(x) = 0 -> nan (pattern 3/4). True pdf ~ c phi^c/x^(c-1), normal for small c.
   Also c > 1: phi(x) subnormal / ndtr(-x)**(c-1) underflow while product normal? (only c ~ 1).
H2 pdf, left tail x << 0, large c: ndtr(-x) = 1 - Phi(x) rounded; **(c-1) amplifies (pattern 6), ~c ulp.
H3 logpdf (l.9016) x >> 0, small c: -x^2/2 and (c-1)*log_ndtr(-x) ~ +x^2/2 cancel -> ~1/c ulp (pattern 5).
H4 logpdf x > 1.34e154: x**2 overflows in _norm_logpdf (INDEX item 2) -> -inf (+ inf) = nan / -inf
   while true -c x^2/2 finite for c < 1.
H5 sf (l.9025) exp(c*log_ndtr(-x)) near x = 0 with large c: cond wrt x ~ 0, error ~ |logsf| ulp
   (exp amplification of rounding in logsf, rounded log_ndtr(0)).
H6 cdf/logsf left tail: log_ndtr(z > 0) carries ndtr(-z) error (shared, lognorm report).
H7 ppf (l.9022) `-ndtri((1-q)**(1/c))`: q < eps -> 1 - q = 1 -> -inf (pattern 8, loss).
   small c: (1-q)**(1/c) underflows -> +inf (loss). large c: argument near 1 -> ndtri cancels.
H8 isf (l.9031) `-ndtri(exp(log(q)/c))`: small c -> exp underflow -> +inf (loss);
   large c or q near 1 -> argument near 1, ndtri loses (pattern 8).
H9 logcdf inherited log(cdf): fine in left tail (cdf = -expm1(tiny) accurate), right tail cdf -> 1 -> 0
   where log1mexp(logsf) works.
