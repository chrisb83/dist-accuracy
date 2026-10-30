# moyal hypotheses (z = exp(-x/2)/sqrt(2))
- pdf L7664: exp(-0.5*(x+exp(-x))): rounding of exp(-x) amplified by |arg| (pattern 6-like) for x << 0; but cond ~ 0.5|x|e^-x as large -> expect ok.
- pdf x>~1416: exp(-x/2) near subnormal; result ~0.4*that -> true pdf subnormal soon after; check window.
- cdf L7667: erfc(z), z large for x<0: erfc cond 2z^2 vs cond(x) |x| z^2; near x in (-2,0) check. x < -7.3 cdf underflows (true too).
- sf L7670: erf(z), z subnormal for x > ~1417.5 while sf = 1.128 z may still be normal (pattern 4), narrow window.
- ppf L7673: -log(2 erfcinv(q)^2): zero crossing at q = erfc(1/sqrt2)=0.31731 (log of ~1) -> cond also infinite there; check. q->1: erfcinv(1-d) with d rounding (pattern 8) - conditioning covers. q tiny/subnormal: erfcinv accuracy (shared).
- inherited: logpdf=log(pdf) -inf for x<~-7.3 (closed form exists); logcdf=log(erfc) underflow x<-7.3 (log(erfcx(z))-z^2); logsf=log(erf) underflow x>~1417 (log(erf z) ~ log(2/sqrtpi)+log z); isf=ppf(1-q) loses for q<1e-16 (-log(2 erfinv(q)^2)).
