# bradford hypotheses
- H1 (p4) cdf: c*x subnormal/underflows for tiny c*x (c ~1e-300, x tiny) -> loss of bits, 0 where cdf ~ x normal.
- H2 (p4/p3) ppf: q*log1p(c) underflows for tiny c and tiny q; then /c. Result ~ q normal.
- H3 (p9) pdf/cdf/ppf small c limit: log1p(c) ~ c, c/c fine; expm1(qL)/c fine. Expect ok.
- H4 large c (1e300): c*x+1 fine; pdf c/(cx+1) ~ 1/x fine. c=inf invalid anyway.
- H5 inherited sf = 1 - cdf: cancels for x -> 1 (pattern 1/12). Better: log1p(c*(1-x)/(1+c*x))/log1p(c).
- H6 inherited isf = ppf(1-q): small q lost. Better: 1 - (1+c)/c * (-expm1(-q*log1p(c))) or expm1((1-q)L)/c wait 1-q rounding: use x = ((1+c)*exp(-qL) - 1)/c.
- H7 logsf inherited: log(sf) above median, sf cancels near x=1.
- ppf near q=1, large c: expm1(qL) with qL large: error ~ qL ulp but cond(q) ~ qL too.
