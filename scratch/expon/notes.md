# expon hypotheses
- cdf (l.2091) -expm1(-x): sc.expm1 inexact for subnormal args (truncexpon F4, pattern 4/10). For expon cdf(x)~x is then subnormal -> not measurable. Check smallest normal x (2.2e-308..1e-300).
- pdf/sf exp(-x): result subnormal for x>708.4 -> not measured. Expect <=1 ulp + cond (cond = x).
- logpdf/logsf -x: exact.
- ppf (l.2094) -log1p(-q): exact up to cond; q near 1 ill-conditioned wrt q only.
- isf (l.2103) -log(q): q near 1, log(q) small; cond 1/|log q| large -> fine wrt arg.
- inherited logcdf = log(cdf): pattern 1/2, cdf rounds to 1 for x > ~37 -> 0.0 (true -exp(-x)). Better: sc.log1p(-np.exp(-x)) / _log1mexp(-x).
