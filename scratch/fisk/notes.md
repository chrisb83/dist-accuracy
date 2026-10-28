# fisk hypotheses (d = 1 specialization of burr)

fisk delegates every method to burr._xxx(x, c, 1.0). So burr findings apply
with d = 1; some vanish (d-dependent), some survive.

H1 (pattern 1) sf/logsf: logsf = log1p(-(1+x**-c)**-1). cdf -> 1 when
   u = x**-c < eps/2 -> log1p(-1) = -inf, sf = 0. True sf ~ x**-c.
   Regime: x > eps**(-1/c). Expect loss.
H2 (pattern 3) left tail x**-c = inf for c*log x < -709.8:
   - pdf: inf/inf = nan; true pdf ~ c*x**(c-1), normal while
     (c-1)log x > -709  => window needs c < 2366 and c > 1.027 to be
     reachable within x >= 1e-300. Use c = 1.2.
   - logpdf: xlog1py(2, inf) = inf -> -inf; true finite.
   - logcdf: -log1p(inf) = -inf; true ~ c*log x (a normal double for any c).
   - cdf: -> 0 but true cdf ~ x**c is subnormal exactly when x**-c overflows
     (d = 1), so probably not measurable. sf -> 1.0 exactly, true 1-x**c
     rounds to 1. Expect no finding for cdf/sf here (unlike burr small d).
H3 (pattern 5) logpdf at c*d - 1 = 0, i.e. c = 1 exactly (and c near 1):
   xlogy(-c-1,x) and xlog1py(2, x**-c) both ~ (c+1)|log x| and cancel.
   Expect total loss at c = 1, partial for c = 1 + 1e-10.
H4 (pattern 4) pdf: x**(-c-1) subnormal while c*x**(-c-1) normal needs
   large c; c = 1e6, x ~ exp(709/c) ~ 1.0007. Expect severe.
H5 (pattern 8) ppf = (q**-1 - 1)**(-1/c): 1/q - 1 cancels for q -> 1, but
   the relative error eps/(1-q) matches cond(q) ~ 1/(c(1-q)); for large c
   it may exceed it (extra 1/c). Check q = 1-1e-16 ... 1-1e-3 with c = 1e6.
   No overflow of q**(-1/d) since d = 1 (1/q finite for q > 1e-308).
H6 (pattern 8) isf: expm1(-log1p(-q)) = q/(1-q), accurate; overflow needs
   1-q < 1e-308 (unreachable). Expect fine (burr F6 was d-driven).
H7 (pattern 6) (1+u)**(d+1) with d=1: benign (power 2 / -1).
H8 (pattern 9) tiny c: ppf = (1/q-1)**(-1/c) underflow/overflow; true value
   also under/overflows -> probably not measurable. Check c = 1e-3.
