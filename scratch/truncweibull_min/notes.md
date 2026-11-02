# truncweibull_min hypotheses (shapes c, a, b; A=a**c, B=b**c, X=x**c)

H1 (p1/p5) all methods: denum exp(-A)-exp(-B) cancels when B-A small relative to 1
   (small a,b; narrow [a,b] with A small; tiny c making A~B~1). rel err ~ eps/(B-A), beyond cond(arg).
   loss when b**c < eps (a=0): denum=0 -> pdf inf, cdf nan.
H2 (p3/p4) all: exp(-A) underflows for A > 745 -> denum 0 -> nan/inf; logpdf +inf. Large a.
H3 (p1) cdf/logcdf a=0 small x: 1-exp(-X) cancels -> 0 for X<eps. near x=a generally: eps/(X-A) vs cond x/(x-a): beyond by 1/(c A).
H4 (p1) sf/logsf near x=b: exp(-X)-exp(-B) cancels, beyond cond by ~1/(c B) when B small.
H5 (p2/p8) ppf small q: -log((1-q)e^-A + q e^-B) -> log of ~1; q<eps -> 0. isf q->1 same.
H6 ppf/isf: both exp underflow (A>745) -> log(0) -> inf.
H7 (p6) exp(-pow(a,c)) with A large: ~A/2 ulp in denum, beyond cond(arg). minor.
H8 logcdf/logsf: log of underflowed num where true log finite (e.g. A large with x slightly above... ) covered by H2.
