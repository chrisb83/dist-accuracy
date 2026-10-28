# dpareto_lognorm hypotheses (SciPy 1.18.1, lines 1837-1969)

z = (log x - u)/s, x1 = a s - z, x2 = b s + z; _logR(t) = log_ndtr(-t) + t^2/2 + log sqrt(2pi)

H1 (p5) logpdf/pdf, logcdf/cdf: logphi(z) ~ -z^2/2 and logR(x1 or x2) ~ +x^2/2 cancel to a
    term linear in z (power-law tails). Error ~ eps*z^2/2 absolute. Regime: |z| large,
    i.e. small s or far tails. Worst for small s (z up to 700/s).
H2 (p1) logsf = log1mexp(logcdf): right tail, logcdf -> -sf; if logcdf is formed as
    log(Phi - c) with Phi ~ 1 then sf < eps -> logsf = -inf; sf = exp(logsf) too.
    Also relative error of sf inherits abs error of logcdf.
H3 (p5) logcdf: logsumexp with signs t3 - t4 (b R(x1) - a R(x2)) cancels where
    b R(x1) ~ a R(x2); and Phi(z) - correction cancel in some regimes (right tail).
H4 (p5) z = (log x - u)/s: abs error eps*|log x| ~ eps*|u| when u large; amplified by 1/s.
H5 large a, b (-> lognormal): a*s - z large; R(x1) ~ 1/x1; fine?
H6 x==0 / inf handled; logcdf at x=inf? out not set -> check.
