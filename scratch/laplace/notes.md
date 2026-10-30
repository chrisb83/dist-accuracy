# laplace hypotheses
- cdf x>0: 1 - 0.5*exp(-x) (pattern 1): result in [0.5,1], no cancellation relative to result. Expect fine.
- sf = cdf(-x): fine by symmetry.
- cdf np.where: exp(x) overflow for x>709 in unused branch (errstate over ignored) -> no warning expected; check.
- ppf q>0.5: -log(2*(1-q)); 1-q exact (Sterbenz) -> fine. q near 0.5: log(2q) with exact 2q -> fine.
- ppf np.where evaluates log(2*(1-q)) for q<=0.5 (fine) and log(2q) for q>0.5 (fine). No warnings expected in (0,1).
- isf = -ppf(q): same.
- inherited logpdf/logcdf/logsf: underflow for |x| > ~745 (pattern 12).
