# truncpareto hypotheses (SciPy 1.18.1, lines 10501-10609)

Normaliser D = 1 - 1/c**b appears in pdf, cdf, sf, ppf, isf, logcdf(b>0), logsf(b>0).

H1 (p2) all methods except logpdf(b>0): D = 1 - c**-b cancels when |b| log c small
    (tiny |b|, or c -> 1). error ~ eps/(|b| log c). Regime b=1e-8, c=5; b=2, c=1+1e-8.
H2 (p1) cdf/logcdf: 1 - x**-b near x = 1 (lower endpoint): err ~ eps/(|b|(x-1)),
    cond wrt x ~ 1 -> severe/loss for x = 1 + tiny.
H3 (p1/p5) sf/logsf: x**-b - c**-b near x = c: err ~ cond/|b|; beyond cond for small |b|.
H4 (p3) b < 0: c**b underflows / 1/c**b overflows when |b| log c > 709 -> pdf, cdf, sf
    give 0/nan; generic logpdf/logcdf/logsf (log of these) -> -inf/nan. b=-1e3,c=5; b=-2,c=1e200.
H5 (p4) b > 0 large: x**-(b+1) underflows/subnormal while b*x**-(b+1) normal (pdf);
    logsf_pos_b: log(x**-b - c**-b) -> -inf when x**-b underflows. b=1e3.
H6 (p6/p8) ppf/isf: pow(base, -1/b): rounding of base amplified by 1/|b| (small b).
H7 (p8) ppf q -> 1 with b log c large: 1 - D*q cancels (D ~ 1). isf near q->1 similar?
    isf: c**-b + D*q fine for small q.
H8 b < 0 logpdf/logcdf/logsf go through rv_continuous generic log(pdf) etc. -- the
    "implemented" methods dispatch to super() for b<0: measure them, they are the code path.
