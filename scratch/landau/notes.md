# landau hypotheses

All five methods are one-line calls into scipy.special._ufuncs._landau_* (Boost.Math
landau_distribution, Yoshimura's piecewise rational/minimax approximations in
transformed variables, ~double precision claimed). No Python-level patterns 1-11.

- H1 (pattern 10/7): Boost uses piecewise approximations on intervals of x;
  far left tail (x < ~-3, pdf ~ exp(-exp(...)) double-exponential) may be
  approximated by a fit with only absolute/limited relative accuracy. Prototype:
  pdf(-5) differs by 1.3e-4 rel; ppf(1e-300) by 2.3e-8 rel.
- H2: interval boundaries of the piecewise fits (unknown exactly) -> add a dense
  x grid in [-5.1, 10] and q grid; look for jumps.
- H3 (pattern 1/8): ppf near q -> 1 and isf near q -> 1: does Boost use the
  complement (isf(1-q)) internally? test q = 1 - 1e-14.
- H4: right tail x huge: pdf ~ 2/(pi x^2): check x up to 1e154 (subnormal after),
  sf ~ 2/(pi x) up to 1e300; isf(1e-300) ~ 6.4e299.
- H5 (doc): docstring pdf is Landau's original standardization, not the Boost one
  SciPy implements: f_scipy(x) = (pi/2) f_L(pi x/2 + log(pi/2)).
