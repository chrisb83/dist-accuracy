# uniform hypotheses
- _pdf/_cdf/_ppf are identities: exact; nothing to fail.
- Inherited sf = 1 - cdf (pattern 1/12): x >= 0.5 exact (Sterbenz); x < 0.5 one rounding -> correctly rounded. Expect fine.
- Inherited isf = ppf(1 - q) (pattern 8): same argument, expect fine.
- Inherited logsf/logcdf: rv_continuous uses log1p(-cdf)/log1p(-sf) beyond the median -> expect fine (logsf(1e-17) = -1e-17 checked).
- loc/scale (public wrapper): z = (x - loc)/scale, ppf = q*scale + loc.
  * x - loc exact near loc (Sterbenz); overflow of x - loc only outside the support (spurious warning).
  * q*scale + loc cancels when loc < 0 < loc + scale and ppf ~ 0: one rounding of q*scale amplified; within cond wrt q.
  * 1/scale overflows for scale < 5.6e-309: true value also beyond double range.
Result: harness 0 ulp everywhere; loc/scale points in locscale.py.
