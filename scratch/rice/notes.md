# rice hypotheses
- H1 pdf (pattern 3/10): x*b overflows to inf for x ~ b >= 1.34e154 -> i0e(inf)=0, pdf 0 (true ~0.4). Probe: rice._pdf(1e155,1e155)=0.0.
- H2 pdf (pattern 5): x-b rounding / exp of (x-b)^2/2 ~ b^2/2: up to ~b^2/2 ulp for small x where cond ~1.
- H3 cdf (ncx2 F7, Ding band): chndtr(x^2,2,b^2) with (x^2/2)^2 subnormal, x ~ 1e-81..1e-77. Probe cdf(1e-80,1e-5) = 4.99994e-161.
- H4 cdf (ncx2 F1): b >= 14.14 (nc >= 200) left tail returns 0. Probe cdf(1e-3,15)=0.
- H5 ppf (ncx2 F6/F7/F8): constant for small q at b>=14.14; F7 band; q->1 excess.
- H6 cdf x^2 underflow for x < 1.5e-154: cdf subnormal there anyway -> not a finding.
- H7 b=0 / b^2 underflow: chndtr(nc=0) works (probe 0.393469 for cdf(1,0)).
- Inherited: sf=1-cdf (right tail loss; better scu._ncx2_sf(x^2,2,b^2)), isf=ppf(1-q) (scu._ncx2_isf), logpdf=log(pdf) (log(x) - (x-b)^2/2 + log(i0e(xb))), logcdf, logsf.
