# logistic hypotheses
- pdf = exp(logpdf): abs error of logpdf (~|x| eps) becomes relative error in pdf; cond(pdf, x) ~ |x|, so expect within conditioning (pattern 3/5 like). Test |x| up to 745.
- logpdf: y - 2*log1p(exp(y)), y=-|x|. No cancellation (both terms negative). Expect fine. Near x=0: -log 4 fine.
- cdf/sf: expit(+-x): check sc.expit relative accuracy in tails (x -> -745), subnormal edge.
- logcdf/logsf: log_expit; check for x near 0 and large |x|, and the -log1p(exp(-x)) branch near x ~ 30..40.
- ppf/isf: logit(q); near q=0.5 result ~4(q-0.5): logit must use log1p form (pattern 2/8). cond large there anyway. q near 1: 1-q rounding, cond ~ 1/(1-q)/log. isf = -logit(q): small q fine.
- pattern 11: none (no literals).
