# truncnorm hypotheses

Z = mass(a, b) = Phi(b) - Phi(a), computed in `_log_gauss_mass` (10219-10256):
left case (b <= 0) logdiff(logPhi(b), logPhi(a)); right (a > 0) mirrored;
central log1p(-Phi(a) - Phi(-b)).

H1 (p5/p7) narrow interval: mass of [lo, hi] with hi - lo small loses
    ~ |logPhi| Phi / (phi (hi-lo)) ulp (left/right) or 1/mass (central).
    -> pdf/logpdf for narrow (a, b): (1, 1+1e-8), (-1e-6, 1e-6).
    -> cdf/logcdf for x near a (mass(a, x)), sf/logsf for x near b (mass(x, b)).
       Beyond cond (x/(x-a)) when a ~ 0: (0, inf) x tiny, (-1e-3, 10) x tiny,
       (-inf, 0) sf for x near 0-.
H2 (p8) ppf/isf via ndtri_exp(log Phi(x)): x has only absolute accuracy eps,
    so relative error eps/|x| when the quantile is close to 0 but cond ~ 1:
    a ~ 0 (ppf small q), b ~ 0 (isf small q).
H3 (p1/p8) isf_right (b >= 0): Phi(-x) = Phi(-a) - (1-q) Z by _log_diff
    (cancellation) instead of Phi(-b) + q Z. Small q with Phi(-b) << Phi(-a):
    b = inf -> log1p(-q) + mass == mass -> -inf -> isf = inf. (-1, inf), (0, inf), (30, inf).
    ppf uses _log_sum (additions) in both branches: expected fine.
H4 logcdf/logsf threshold -0.1: both sides; expected fine.
H5 far tails (30, inf), (-50, -10): log masses ~ -a^2/2 cancel in
    logcdf = mass(a,x) - mass(a,b): abs error eps a^2/2; cond(arg) also ~ a^2.
