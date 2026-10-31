# recipinvgauss hypotheses
X = 1/Y, Y ~ invgauss(mu). cdf_X(x) = sf_Y(1/x). z1 = (1/mu - x)/sqrt x, z2 = (1/mu + x)/sqrt x, z2^2 - z1^2 = 4/mu.
exp(2/mu) phi(z2) = phi(z1) exactly.

H1 (p5/p1) cdf L9576: Phi(-z1) - exp(2/mu)Phi(-z2) cancels for x << 1/mu: ratio of terms -> z1/z2, loss ~ (1/mu + x)/(2x) ulp. Left tail all mu.
H2 (p3) cdf/sf: exp(2/mu) overflows for mu < 2/709.78 = 2.818e-3 -> inf*Phi = inf/-inf, inf*0 = nan. Everywhere.
H3 (p4) cdf/sf: Phi(-z2) underflows (z2^2/2 > ~745) while exp(2/mu)*Phi(-z2) normal (small mu, e.g. 0.01): sf right tail loses second term (rel error ~ |z1|/(|z1|+z2)); cdf left tail loses subtrahend -> loss.
H4 (p5) cdf/sf small mu near x~1/mu: exp(2/mu) relative error ~ (2/mu) eps; ndtr(-z2) arg error ~ z2^2 eps ~ 4/mu ulp; plus sc.ndtr far-tail error (INDEX 6, up to ~1000 ulp for |z| in 20..37).
H5 (p3) logpdf L9568: mu**2 overflows for mu > 1.34e154 (inf/inf nan or finite/inf = 0 -> wrong); (1 - mu*x)**2 overflows for mu*x > 1.34e154; mu**2 underflows for mu < 1.5e-154 -> /0 = -inf or 0/0 nan. pdf = exp(logpdf) inherits.
H6 pdf = exp(logpdf): amplification up to |logpdf| ulp (<= ~745), probably within cond.
H7 sf right tail: both terms positive, ok for moderate mu.
