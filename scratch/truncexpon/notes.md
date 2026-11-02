# truncexpon hypotheses
- H1 (p1/p5) sf L10184: exp(-b) - exp(-x) cancels when b small (both ~1): rel err ~eps/b at all x; b < 1.1e-16 -> 0.
- H2 (p2) logpdf L10175: log(-expm1(-b)) for large b: -expm1(-b) rounds to 1 -> log = 0 vs -e^-b; logpdf at x << e^-b loses everything (b > ~37), severe for b ~ 20 at x=0.
- H3 (p2/p8) isf L10187: -log(exp(-b) - q*expm1(-b)) for small b: argument ~ 1 - b(1-q); log of near-1 loses eps/b relative.
- H4 (p8) ppf L10181: q->1 large b: cancellation in log1p(q*expm1(-b)), expected within conditioning.
- H5 sf with x near b: cancellation but cond wrt x ~ x/(b-x); beyond conditioning only for small b (H1).
- pdf, cdf: expect fine.
