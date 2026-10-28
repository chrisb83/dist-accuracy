# cauchy hypotheses
- pdf (l.1478) 1/pi/(1+x*x): x*x overflows for |x|>1.34e154, but true pdf there is subnormal -> not a finding. Pattern 3, no expected effect.
- logpdf (l.1490-1493) branch at |x|=1; large-|x| branch avoids overflow. Check both sides of 1, |x| up to 1e308. Expect fine.
- cdf/sf (l.1496/1502) arctan2(1,-x)/pi: tails fine (atan(1/|x|)); no 1-F. Expect fine; check x near 0 and tails.
- ppf/isf (boost _cauchy_ppf/_cauchy_isf): -1/tan(pi*p) with p reflected; near q=0.5 cond ~0.5/|q-0.5| (pattern 8). Check q around 0.5, q near 1 (1-q exact by Sterbenz?), tiny q (1/(pi q) overflows for q<1.8e-309, fine).
- inherited logcdf/logsf: log(cdf) left tail where cdf subnormal (x < -1.4e307) -> degraded; switch at median uses log1p(-sf). Opportunity only.
