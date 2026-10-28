# mielke hypotheses (code reading, SciPy 1.18.1, lines 7193-7252)

Code (standard form):

```
_pdf   : k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)          line 7232-7233
_logpdf: log(k) + log(x)*(k-1) - log1p(x**s)*(1 + k/s)     line 7235-7238
_cdf   : x**k / (1.0+x**s)**(k*1.0/s)                      line 7240-7241
_ppf   : qsk = pow(q, s/k); pow(qsk/(1.0-qsk), 1.0/s)      line 7243-7245
```

mielke(k, s) == burr(c = s, d = k/s):
cdf = (1 + x**-s)**(-k/s), sf = -expm1(-(k/s)*log1p(x**-s)).

| # | pattern | method | regime where it should fail |
|---|---|---|---|
| H1 | 3 (overflow of intermediate that cancels) | cdf | x**k overflows for k*log x > 709.8 AND (1+x**s)**(k/s) ~ x**k overflows too -> inf/inf = nan, true cdf ~ 1. Right tail, any k; threshold x > exp(709.8/k) |
| H2 | 3 | pdf | denominator (1+x**s)**(1+k/s) ~ x**(s+k) overflows at (s+k)log x > 709.8 while numerator x**(k-1) still finite -> finite/inf = 0, true pdf ~ k x**(-s-1) normal. Also numerator overflow first if k-1 > s+k (impossible) ; for k>1 both can overflow -> nan |
| H3 | 3 | logpdf | x**s overflows for s*log x > 709.8 -> log1p(inf) = inf -> -inf although logpdf finite |
| H4 | 5 (cancellation in log space) | logpdf | x >> 1: (k-1)log x - (s+k)log x, result -(s+1)log x. loss factor ~ (s+k)/(s+1), big when k >> s (k=1e6, s=1e-3) |
| H5 | 1/8 | ppf | 1.0 - qsk cancels when qsk -> 1: q -> 1, or s/k << 1. Error then amplified by 1/s power (pattern 6) when s < 1 |
| H6 | 6 (rounding raised to power) | cdf, pdf | k/s and 1+k/s are rounded doubles; exponent error * |log x| ; also (1+x**s)**big multiplies the 1/2 ulp of 1+x**s by k/s -> large shapes |
| H7 | 4 (subnormal intermediate) | pdf | x**(k-1) subnormal while k*... lifts result back; large k with x slightly < 1 |
| H8 | 3 | cdf | small x: x**k underflows to 0 for k*log x < -745 while true cdf ~ x**k also underflows -> not a finding (true value subnormal) |
| H9 | inherited | sf/logsf/logcdf/isf | generic sf = 1-cdf collapses in right tail; cdf itself nan there (H1) -> sf nan |
