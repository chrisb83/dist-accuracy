# arcsine hypotheses
- pdf L638 `1.0/np.pi/np.sqrt(x*(1-x))`: 1-x exact for x>=0.5, rel. eps for x<0.5; expect few ulp everywhere. Tiny x: x*(1-x) ok down to subnormal x.
- cdf L641 `2/pi*arcsin(sqrt(x))`: expect fine; near x=1 cdf ~ 1, well conditioned.
- ppf L644 `sin(pi/2*q)**2`: small q: rel fine until result subnormal (q < ~1e-154). q near 1: sin flat, fine.
- inherited sf = 1 - cdf (pattern 1, x -> 1), isf = ppf(1-q) (pattern 8, small q), logsf = log(1-cdf) for x>0.5.
- logcdf default: log1p(-(1-cdf)) for x>0.5 -- fine.
